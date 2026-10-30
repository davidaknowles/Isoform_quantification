"""Tests for the long-read truncation model and PAF-based EC construction."""

import numpy as np
import pandas as pd

from joli.joint import JointEM
from joli.lr_bias import TruncationModel, fit_with_truncation
from joli.paf import filter_alignments, library_from_alignments


def simulate_nested(rng, n_genes=200, n_reads=40000):
    """
    Genes with a long isoform (2000-4000 bp) and a short isoform equal to its
    3' end (500-1500 bp). Reads are 3'-anchored; the 5' start is drawn from a
    profile where 60% of reads are full length and the rest truncated
    uniformly. Returns alignment table, transcript lengths, true theta.
    """
    names, lens = [], []
    for g in range(n_genes):
        L = rng.integers(2000, 4000)
        S = rng.integers(500, 1500)
        names += [f"g{g}_long", f"g{g}_short"]
        lens += [L, S]
    lens = np.array(lens)
    T = len(names)
    theta = rng.dirichlet(np.ones(T) * 0.5)
    tx = rng.choice(T, size=n_reads, p=theta)
    full = rng.random(n_reads) < 0.6
    start = np.where(full, 0, (rng.random(n_reads) * lens[tx]).astype(int))
    rows = []
    for r, (t, s) in enumerate(zip(tx, start)):
        g = t // 2
        L, S = lens[2 * g], lens[2 * g + 1]
        # position in long-isoform coordinates
        s_long = s if t % 2 == 0 else s + (L - S)
        rows.append((r, 2 * g, s_long, L, L))
        if s_long >= L - S:                       # read lies within the short isoform too
            rows.append((r, 2 * g + 1, s_long - (L - S), S, S))
    aln = pd.DataFrame(rows, columns=["read", "tx", "tstart", "tend", "tlen"])
    aln["tx"] = np.array(names)[aln["tx"]]
    aln["AS"] = 100.0
    return aln, names, lens, theta


def test_truncation_model_improves_nested_isoforms():
    rng = np.random.default_rng(0)
    aln, names, lens, theta = simulate_nested(rng)
    aln = filter_alignments(aln)
    plain = library_from_alignments("lr", aln, names)
    res0 = JointEM([plain]).run()
    model = TruncationModel(lens)
    lib = library_from_alignments("lr", aln, names, features=model.features(aln["tstart"], aln["tend"], aln["tlen"]))
    em = JointEM([lib])
    res1 = fit_with_truncation(em, {0: model}, n_outer=10)
    err0 = np.abs(res0.theta[0] - theta).sum()
    err1 = np.abs(res1.theta[0] - theta).sum()
    assert err1 < 0.5 * err0, (err0, err1)
    # 3' ends are always at the transcript end: the d3 = 0 bin dominates
    assert model.end3.p[0] > 0.95


def test_rich_ec_weights_and_features_collapse():
    aln = pd.DataFrame({"read": ["a", "a", "b", "b", "c"], "tx": ["t1", "t2", "t1", "t2", "t1"],
                        "AS": [10.0, 9.9, 10.0, 10.0, 10.0]})
    lib = library_from_alignments("x", aln, ["t1", "t2"], weight=np.array([1.0, 3.0, 1.0, 1.0, 2.0]))
    assert sorted(lib.tcc.ec_counts.tolist()) == [1.0, 2.0]
    two = [i for i, t in enumerate(lib.tcc.ec_transcripts) if len(t) == 2][0]
    # mean of per-read normalised weights: ((0.25, 0.75) + (0.5, 0.5)) / 2
    np.testing.assert_allclose(lib.c[two], [0.375, 0.625])
    feat = np.array([[0], [1], [0], [0], [0]])
    lib2 = library_from_alignments("x", aln, ["t1", "t2"], features=feat)
    assert len(lib2.tcc.ec_counts) == 3


def test_length_bias_recovered_with_reference_library():
    """Two LR libraries share theta; library 1 samples transcripts at rate L^-0.5."""
    from joli.joint import Library
    from joli.load_tcc import TCCData
    from joli.lr_bias import LengthBias, fit_bias
    rng = np.random.default_rng(1)
    T = 400
    L = rng.integers(300, 10000, size=T).astype(float)
    theta = rng.dirichlet(np.ones(T))
    libs = []
    for slope in [0.0, -0.5]:
        p = theta * L ** slope
        n = rng.multinomial(200000, p / p.sum()).astype(float)
        tcc = TCCData(ec_counts=n, ec_transcripts=[np.array([t]) for t in range(T)],
                      transcript_names=[f"t{t}" for t in range(T)], total_reads=int(n.sum()),
                      n_targets=T, sample_dir="")
        libs.append(Library(name=f"l{slope}", tcc=tcc, b=np.ones(T)))
    em = JointEM(libs, units=[0, 0])
    lb = LengthBias(L, knots=(), ridge=1e-3)
    res = fit_bias(em, length={1: lb}, n_outer=10)
    assert abs(lb.w[0] + 0.5) < 0.03, lb.w
    np.testing.assert_allclose(res.theta[0], theta, atol=5e-3)


def test_shared_models_pool_across_units():
    """Two units on the same genes, one long-read library each, sharing one truncation model."""
    from joli.lr_bias import fit_bias
    aln0, names, lens, th0 = simulate_nested(np.random.default_rng(3), n_reads=30000)
    # unit 1: other abundances on unit 0's genes, same read-end process
    rng1 = np.random.default_rng(5)
    th1 = rng1.dirichlet(np.ones(len(names)) * 0.5)
    tx = rng1.choice(len(names), size=30000, p=th1)
    full = rng1.random(30000) < 0.6
    start = np.where(full, 0, (rng1.random(30000) * lens[tx]).astype(int))
    rows = []
    for r, (t, s) in enumerate(zip(tx, start)):
        g = t // 2
        L, S = lens[2 * g], lens[2 * g + 1]
        s_long = s if t % 2 == 0 else s + (L - S)
        rows.append((r, names[2 * g], s_long, L, L))
        if s_long >= L - S:
            rows.append((r, names[2 * g + 1], s_long - (L - S), S, S))
    aln1 = pd.DataFrame(rows, columns=["read", "tx", "tstart", "tend", "tlen"])
    aln1["AS"] = 100.0
    model = TruncationModel(lens)
    libs = [library_from_alignments(f"u{u}", filter_alignments(a), names,
                                    features=model.features(*(filter_alignments(a)[c] for c in ("tstart", "tend", "tlen"))))
            for u, a in enumerate([aln0, aln1])]
    res = fit_bias(JointEM(libs, units=[0, 1]), trunc={0: model, 1: model}, n_outer=10)
    for u, th in enumerate([th0, th1]):
        assert np.abs(res.theta[u] - th).sum() < 0.15
    assert model.end3.p[0] > 0.95


def test_unspliced_internal_priming_scale_recovered():
    """poly(dT): rate 1 on spliced targets, kappa * L on unspliced; hexamer: L everywhere."""
    from joli.alevin import primer_rates
    from joli.joint import Library
    from joli.load_tcc import TCCData
    from joli.lr_bias import LengthBias, fit_bias
    rng = np.random.default_rng(7)
    T = 600
    L = rng.integers(500, 50000, size=T).astype(float)
    unspliced = np.arange(T) >= 400
    theta = rng.dirichlet(np.ones(T))
    kappa = 0.01
    true_rates = {"poly(dT)": np.where(unspliced, kappa * L, 1.0), "random hexamer": L}
    libs = []
    for p in ["poly(dT)", "random hexamer"]:
        pr = theta * true_rates[p]
        n = rng.multinomial(300000, pr / pr.sum()).astype(float)
        tcc = TCCData(ec_counts=n, ec_transcripts=[np.array([t]) for t in range(T)],
                      transcript_names=[f"t{t}" for t in range(T)], total_reads=int(n.sum()), n_targets=T, sample_dir="")
        libs.append(Library(p, tcc, b=primer_rates(p, L, 0, unspliced)))
    lb = LengthBias(L, extra=unspliced[:, None], length_terms=False, ridge=1e-3)
    res = fit_bias(JointEM(libs, units=[0, 0]), length={0: lb}, n_outer=10)
    r = lb.rates()
    est = r[unspliced].mean() / r[~unspliced].mean()        # learned multiplier on the base rate L
    assert abs(np.log(est) - np.log(kappa * L[unspliced].mean() / (L[unspliced] + 1).mean())) < 0.1, est
    np.testing.assert_allclose(res.theta[0], theta, atol=5e-3)


def test_fragment_model_corrects_3prime_coverage_bias():
    """
    Genes with a long isoform (unique 5' exon + shared body) and a short isoform
    (shared body only). SR fragment starts are 3'-biased (u ~ Beta(3, 1));
    an unbiased LR library shares theta. The fragment model learns q and fld
    and improves theta over a uniform-coverage model.
    """
    from joli.joint import JointEM, Library
    from joli.load_tcc import TCCData
    from joli.lr_bias import FragmentModel, fit_bias
    rng = np.random.default_rng(11)
    G = 150
    names, lens, offs = [], [], []
    for g in range(G):
        A, B = rng.integers(300, 1500), rng.integers(800, 2500)
        names += [f"g{g}_long", f"g{g}_short"]
        lens += [A + B, B]
        offs.append(A)
    lens = np.array(lens, dtype=float)
    T = len(names)
    theta = rng.dirichlet(np.ones(T))
    fld_mean = 300
    # SR: molecules ~ theta * eff len, fragment f ~ N(300, 50), start u ~ Beta(3, 1)
    n_sr = 200000
    flen_all = np.clip(rng.normal(fld_mean, 50, size=4 * n_sr).astype(int), 100, 600)
    eff = np.array([np.mean(np.maximum(L - flen_all[:2000] + 1, 0)) for L in lens])
    tx = rng.choice(T, size=n_sr, p=theta * eff / (theta * eff).sum())
    f = flen_all[:n_sr]
    ok = f < lens[tx]
    tx, f = tx[ok], f[ok]
    s = (rng.beta(3, 1, size=len(tx)) * (lens[tx] - f)).astype(int)
    rows = []
    for r, (t, st, fl) in enumerate(zip(tx, s, f)):
        g, A = t // 2, offs[t // 2]
        s_long = st if t % 2 == 0 else st + A
        rows.append((r, 2 * g, s_long, fl))
        if s_long >= A:
            rows.append((r, 2 * g + 1, s_long - A, fl))
    frags = pd.DataFrame(rows, columns=["read", "tx_idx", "start", "flen"])
    model = FragmentModel(lens)
    from joli.salmon_io import library_from_fragments
    sr = library_from_fragments("sr", frags, names, lens, model)
    # LR: one read per molecule, uniquely assigned
    n_lr = rng.multinomial(20000, theta).astype(float)
    lr = Library("lr", TCCData(ec_counts=n_lr, ec_transcripts=[np.array([t]) for t in range(T)], transcript_names=names,
                               total_reads=int(n_lr.sum()), n_targets=T, sample_dir=""), b=np.ones(T), platform="long")
    uniform = FragmentModel(lens); uniform.pf = model.pf.copy()
    em0 = JointEM([Library("sr", sr.tcc, uniform.rates(), "short", feat=sr.feat), lr], units=[0, 0])
    res0 = em0.run()
    res1 = fit_bias(JointEM([sr, lr], units=[0, 0]), frag={0: model}, n_outer=10)
    err0, err1 = np.abs(res0.theta[0] - theta).sum(), np.abs(res1.theta[0] - theta).sum()
    assert err1 < 0.6 * err0, (err0, err1)
    assert model.pq[-5:].sum() > 0.45          # learned 3' bias (Beta(3,1): 0.58 of mass in the last quarter)
    mean_f = (model.pf * model.fmid).sum()
    assert abs(mean_f - fld_mean) < 25
