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
