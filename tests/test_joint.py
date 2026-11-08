"""Tests for joli.joint.JointEM."""

import numpy as np
import pytest

from joli.em_algorithm import JoliEM
from joli.joint import JointEM, Library, molar_to_tpm
from joli.load_tcc import TCCData
from joli.weights import WeightData

SHARED = 280.0   # bp shared by all transcripts of a gene (short reads)


def simulate_tcc(theta, b, genes, rng, n_reads, p_unique):
    """
    Draw reads with P(t) proportional to theta_t b_t. A read is unique to its
    transcript with probability p_unique[t], otherwise compatible with its whole
    gene. For short reads p_unique[t] = 1 - shared / b_t (a shared region of fixed
    length per gene), so every EC has the same number of positions in each of
    its transcripts, as kallisto's likelihood assumes.
    """
    T = len(theta)
    p = theta * b
    t_draw = rng.choice(T, size=n_reads, p=p / p.sum())
    unique = rng.random(n_reads) < np.broadcast_to(p_unique, (T,))[t_draw]
    ec_index = {}
    ec_tx, counts = [], []
    for t, u in zip(t_draw, unique):
        key = (int(t),) if u else tuple(np.flatnonzero(genes == genes[t]).tolist())
        if key not in ec_index:
            ec_index[key] = len(ec_tx)
            ec_tx.append(list(key))
            counts.append(0)
        counts[ec_index[key]] += 1
    names = [f"tx{i}" for i in range(T)]
    return TCCData(ec_counts=np.array(counts, dtype=np.int64), ec_transcripts=ec_tx,
                   transcript_names=names, total_reads=n_reads, n_targets=T, sample_dir="")


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n_genes, per_gene = 60, 4
    genes = np.repeat(np.arange(n_genes), per_gene)
    T = len(genes)
    theta = rng.dirichlet(np.full(T, 0.5))
    eff_len = rng.uniform(300, 4000, size=T)
    return rng, genes, theta, eff_len


def test_single_long_library_matches_joliem(toy):
    rng, genes, theta, eff_len = toy
    tcc = simulate_tcc(theta, np.ones_like(theta), genes, rng, 5000, 0.5)
    lib = Library("lr", tcc, b=np.ones(len(theta)), platform="long")
    res = JointEM([lib]).run(max_iter=3000, min_iter=3000, rel_tol=0.0, accelerate=False)
    w = WeightData(eff_lens=np.ones(len(theta)), ec_weights=[np.ones(len(e)) for e in tcc.ec_transcripts])
    ref = JoliEM(tcc, w, em_include_single_tx=True).run(max_em_rounds=3000, min_rounds=3000,
                                                        convergence_mode="joli")
    ref_theta = ref.theta_unnorm / ref.theta_unnorm.sum()
    big = ref_theta > 1e-4
    np.testing.assert_allclose(res.theta[0][big], ref_theta[big], rtol=1e-3)


def test_single_short_library_matches_kallisto_rho(toy):
    rng, genes, theta, eff_len = toy
    tcc = simulate_tcc(theta, eff_len, genes, rng, 20000, 1 - SHARED / eff_len)
    lib = Library("sr", tcc, b=eff_len, platform="short")
    res = JointEM([lib]).run(max_iter=3000, min_iter=3000, rel_tol=0.0, accelerate=False)
    w = WeightData(eff_lens=eff_len, ec_weights=[1.0 / eff_len[e] for e in tcc.ec_transcripts])
    ref = JoliEM(tcc, w, em_include_single_tx=True).run(max_em_rounds=3000, min_rounds=3000,
                                                        convergence_mode="joli")
    rho = ref.theta_unnorm / eff_len
    rho /= rho.sum()
    big = rho > 1e-4
    np.testing.assert_allclose(res.theta[0][big], rho[big], rtol=1e-3)


def test_loglik_monotone_with_mixed_libraries(toy):
    rng, genes, theta, eff_len = toy
    sr = Library("sr", simulate_tcc(theta, eff_len, genes, rng, 20000, 1 - SHARED / eff_len), b=eff_len, platform="short")
    lr = Library("lr", simulate_tcc(theta, np.ones_like(theta), genes, rng, 2000, 0.8),
                 b=np.ones(len(theta)), platform="long")
    res = JointEM([sr, lr], units=[0, 0]).run(max_iter=300, min_iter=300, rel_tol=0.0, accelerate=False)
    ll = np.array(res.loglik)
    assert np.all(np.diff(ll) > -1e-6 * abs(ll[-1]))


def test_joint_beats_each_platform(toy):
    rng, genes, theta, eff_len = toy
    sr = Library("sr", simulate_tcc(theta, eff_len, genes, rng, 20000, 1 - SHARED / eff_len), b=eff_len, platform="short")
    lr = Library("lr", simulate_tcc(theta, np.ones_like(theta), genes, rng, 1500, 0.9),
                 b=np.ones(len(theta)), platform="long")

    def err(libs):
        th = JointEM(libs, units=[0] * len(libs)).run(max_iter=3000).theta[0]
        return np.abs(molar_to_tpm(th) - molar_to_tpm(theta)).sum()

    e_sr, e_lr, e_joint = err([sr]), err([lr]), err([sr, lr])
    assert e_joint < min(e_sr, e_lr)


def test_mixture_units_recover_components():
    rng = np.random.default_rng(1)
    genes = np.repeat(np.arange(40), 3)
    T = len(genes)
    th = rng.dirichlet(np.full(T, 0.5), size=2)
    pi = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    libs = []
    for l in range(3):
        mu = pi[l] @ th
        libs.append(Library(f"l{l}", simulate_tcc(mu, np.ones(T), genes, rng, 40000, 0.6), b=np.ones(T)))
    est = JointEM(libs, pi=pi).run(max_iter=3000).theta
    for k in range(2):
        assert np.abs(est[k] - th[k]).sum() < 0.15


def test_squarem_reaches_plain_em_optimum(toy):
    rng, genes, theta, eff_len = toy
    sr = Library("sr", simulate_tcc(theta, eff_len, genes, rng, 20000, 1 - SHARED / eff_len), b=eff_len, platform="short")
    lr = Library("lr", simulate_tcc(theta, np.ones_like(theta), genes, rng, 2000, 0.8),
                 b=np.ones(len(theta)), platform="long")
    em = JointEM([sr, lr], units=[0, 0])
    plain = em.run(max_iter=20000, min_iter=20000, rel_tol=0.0, accelerate=False)
    fast = em.run(rel_tol=1e-4)
    assert fast.n_iter < 2000
    assert fast.loglik[-1] >= plain.loglik[-1] - 1e-3
    big = plain.theta[0] > 1e-4
    np.testing.assert_allclose(fast.theta[0][big], plain.theta[0][big], rtol=1e-2)


def test_onehot_estep_matches_dense():
    from joli.joint import JointEM, Library
    from joli.load_tcc import TCCData, FlatECs
    rng = np.random.default_rng(3)
    T, L = 300, 7
    names = [f"t{i}" for i in range(T)]
    libs = []
    for l in range(L):
        sizes = rng.integers(1, 5, size=200)
        ptr = np.r_[0, np.cumsum(sizes)]
        tcc = TCCData(ec_counts=rng.integers(1, 9, 200).astype(float),
                      ec_transcripts=FlatECs(ptr, rng.integers(0, T, ptr[-1]).astype(np.int32)),
                      transcript_names=names, total_reads=0, n_targets=T, sample_dir="")
        libs.append(Library(f"l{l}", tcc, b=rng.uniform(0.5, 2, T), c=FlatECs(ptr, rng.uniform(0.1, 1, ptr[-1]))))
    units = [2, 0, 1, 2, 0, 2, 1]
    fast = JointEM(libs, units=units)
    dense = JointEM(libs, units=units)
    dense._unit_of, dense._row = None, np.arange(L)
    theta = rng.dirichlet(np.ones(fast.T), size=3)
    a, b = fast._estep(theta, expected_reads=True), dense._estep(theta, expected_reads=True)
    for x, y in [(a[0], b[0]), (a[1], b[1]), (a[3], b[3])]:
        np.testing.assert_allclose(x, y, rtol=1e-12)
    assert abs(a[2] - b[2]) < 1e-8 * abs(b[2])
    np.testing.assert_allclose(fast.run().theta, dense.run().theta, atol=1e-10)


def test_library_loglik_matches_fit_loglik():
    from joli.joint import JointEM, Library
    from joli.load_tcc import TCCData, FlatECs
    rng = np.random.default_rng(4)
    T, L = 200, 4
    names = [f"t{i}" for i in range(T)]
    libs = []
    for l in range(L):
        sizes = rng.integers(1, 4, size=150)
        ptr = np.r_[0, np.cumsum(sizes)]
        tcc = TCCData(ec_counts=rng.integers(1, 9, 150).astype(float),
                      ec_transcripts=FlatECs(ptr, rng.integers(0, T, ptr[-1]).astype(np.int32)),
                      transcript_names=names, total_reads=0, n_targets=T, sample_dir="")
        libs.append(Library(f"l{l}", tcc, b=rng.uniform(0.5, 2, T), c=FlatECs(ptr, rng.uniform(0.1, 1, ptr[-1]))))
    em = JointEM(libs, units=[0, 1, 0, 1])
    res = em.run(zero_below=None)
    ll, n_zero = em.library_loglik(res.theta)
    assert abs(ll.sum() - res.loglik[-1]) < 1e-8 * abs(res.loglik[-1])
    assert n_zero.sum() == 0
    # an EC on a transcript with zero abundance: -inf without smoothing, finite with it
    theta = res.theta.copy()
    t0 = libs[0].tcc.ec_transcripts[0]
    theta[0, t0] = 0.0
    theta /= theta.sum(axis=1, keepdims=True)
    ll0, nz0 = em.library_loglik(theta)
    ll1, _ = em.library_loglik(theta, eps=1e-6)
    assert nz0[0] > 0 and np.isneginf(ll0[0]) and np.isfinite(ll1[0])


def test_library_loglik_within_one_gene_equals_conditional():
    """With every transcript in one group, the within-group score is the full log-likelihood."""
    from joli.joint import JointEM, Library
    from joli.load_tcc import TCCData, FlatECs
    rng = np.random.default_rng(5)
    T = 50
    names = [f"t{i}" for i in range(T)]
    sizes = rng.integers(1, 4, size=80)
    ptr = np.r_[0, np.cumsum(sizes)]
    tcc = TCCData(ec_counts=rng.integers(1, 9, 80).astype(float),
                  ec_transcripts=FlatECs(ptr, rng.integers(0, T, ptr[-1]).astype(np.int32)),
                  transcript_names=names, total_reads=0, n_targets=T, sample_dir="")
    em = JointEM([Library("l", tcc, b=rng.uniform(0.5, 2, T))])
    th = em.run().theta
    full, _ = em.library_loglik(th, eps=1e-9)
    within, used, _ = em.library_loglik_within(th, np.zeros(T, dtype=int), eps=1e-9)
    assert abs(full[0] - within[0]) < 1e-8 * abs(full[0]) and used[0] == tcc.ec_counts.sum()
    # two groups: ECs spanning both are excluded
    g = (np.arange(T) >= 25).astype(int)
    _, used2, _ = em.library_loglik_within(th, g, eps=1e-9)
    span = [len(set(g[t])) > 1 for t in tcc.ec_transcripts]
    assert used2[0] == tcc.ec_counts[~np.array(span)].sum()


def _ec_library(name, rng, ecs, probs, n, T, names, b):
    from joli.joint import Library
    from joli.load_tcc import TCCData, FlatECs
    counts = rng.multinomial(n, probs).astype(float)
    ptr = np.r_[0, np.cumsum([len(e) for e in ecs])]
    tcc = TCCData(ec_counts=counts, ec_transcripts=FlatECs(ptr, np.concatenate(ecs).astype(np.int32)),
                  transcript_names=names, total_reads=n, n_targets=T, sample_dir="")
    return Library(name, tcc, b=b)


def _two_platforms(rng, G=150, iso=3, gene_shift=1.0):
    """Genes with `iso` isoforms; platform 0 resolves isoforms (few reads), platform 1 mostly
    gene-level ECs (many reads) and gene expression shifted by lognormal(0, gene_shift)."""
    T = G * iso
    names = [f"t{i}" for i in range(T)]
    gene = np.repeat(np.arange(G), iso)
    psi = rng.dirichlet(np.ones(iso), size=G).ravel()
    expr = rng.lognormal(0, 1, G)
    libs = []
    for p, (n, shift) in enumerate([(4000, 0.0), (200000, gene_shift)]):
        e = expr * np.exp(rng.normal(0, shift, G)) if shift else expr
        theta = e[gene] * psi
        theta /= theta.sum()
        if p == 0:                                    # one EC per transcript
            ecs, probs = [np.array([t]) for t in range(T)], theta
        else:                                         # 80% gene-level ECs, 20% transcript-unique
            ecs = [np.arange(g * iso, (g + 1) * iso) for g in range(G)] + [np.array([t]) for t in range(T)]
            probs = np.r_[0.8 * np.bincount(gene, theta), 0.2 * theta]
        libs.append(_ec_library(f"p{p}", rng, ecs, probs / probs.sum(), n, T, names, np.ones(T)))
    return libs, gene, psi


def test_tied_fit_with_separate_ties_matches_untied():
    from joli.joint import JointEM
    libs, gene, _ = _two_platforms(np.random.default_rng(0))
    a = JointEM(libs, units=[0, 1]).run(rel_tol=1e-8, max_iter=20000, zero_below=None)
    b = JointEM(libs, units=[0, 1], tie=[0, 1], groups=gene).run(rel_tol=1e-8, max_iter=20000, zero_below=None)
    np.testing.assert_allclose(a.theta, b.theta, atol=1e-6)
    assert abs(a.loglik[-1] - b.loglik[-1]) < 1e-6 * abs(a.loglik[-1])


def test_tied_fractions_pool_platforms_when_gene_expression_differs():
    """Gene-level shifts between platforms: tied fractions pool both platforms (as good as a shared
    theta, which is not biased within genes by gene-level shifts) and beat the resolving platform alone."""
    from joli.joint import JointEM
    libs, gene, psi = _two_platforms(np.random.default_rng(1), gene_shift=1.0)
    def within_err(theta):
        tot = np.bincount(gene, theta)
        est = np.divide(theta, tot[gene], out=np.zeros_like(theta), where=tot[gene] > 0)
        return 0.5 * np.abs(est - psi).sum() / len(np.unique(gene))
    shared = JointEM(libs, units=[0, 0]).run(rel_tol=1e-6, max_iter=20000).theta[0]
    tied = JointEM(libs, units=[0, 1], tie=[0, 0], groups=gene).run(rel_tol=1e-6, max_iter=20000).theta[0]
    lr_only = JointEM(libs[:1]).run(rel_tol=1e-6, max_iter=20000).theta[0]
    e_shared, e_tied, e_lr = within_err(shared), within_err(tied), within_err(lr_only)
    assert e_tied <= e_shared + 1e-3 and e_tied < 0.6 * e_lr, (e_shared, e_tied, e_lr)


def _close_theta(a, b, tol):
    return 0.5 * np.abs(a - b).sum(axis=1).max() < tol


def test_pooling_limits():
    """Small tau: pooled units approach one shared theta; large tau: separate units (transcript level)
    or tied units with free gene totals (gene level)."""
    from joli.joint import JointEM
    libs, gene, _ = _two_platforms(np.random.default_rng(2), G=60, gene_shift=1.0)
    kw = dict(rel_tol=1e-7, max_iter=50000, zero_below=None)
    shared = JointEM(libs, units=[0, 0]).run(**kw).theta[0]
    separate = JointEM(libs, units=[0, 1]).run(**kw).theta
    tied = JointEM(libs, units=[0, 1], tie=[0, 0], groups=gene).run(**kw).theta
    for level in ["transcript", "gene"]:
        small = JointEM(libs, units=[0, 1], tie=[0, 0], groups=gene, pool=level, pool_tau=0.003).run(**kw).theta
        assert _close_theta(small, np.stack([shared, shared]), 0.02), level
        large = JointEM(libs, units=[0, 1], tie=[0, 0], groups=gene, pool=level, pool_tau=30.0).run(**kw).theta
        assert _close_theta(large, separate if level == "transcript" else tied, 0.02), level


def test_pooled_fit_increases_log_posterior():
    from joli.joint import JointEM
    libs, gene, _ = _two_platforms(np.random.default_rng(3), G=60, gene_shift=1.0)
    for level in ["transcript", "gene"]:
        res = JointEM(libs, units=[0, 1], tie=[0, 0], groups=gene, pool=level, pool_tau=0.5).run(
            rel_tol=1e-7, max_iter=20000, zero_below=None, accelerate=False)
        lp = np.array(res.loglik[:-1])
        assert (np.diff(lp[1:]) > -1e-6 * abs(lp[-1])).all(), level


def test_pooled_fit_with_thread_pool():
    """Pooled fits on libraries large enough for the threaded E-step (regression: attribute clash)."""
    from joli.joint import JointEM, Library
    from joli.load_tcc import TCCData, FlatECs
    rng = np.random.default_rng(9)
    T = 2000
    names = [f"t{i}" for i in range(T)]
    gene = np.arange(T) // 4
    libs = []
    for l in range(2):
        sizes = rng.integers(1, 4, size=300000)
        ptr = np.r_[0, np.cumsum(sizes)]
        tcc = TCCData(ec_counts=rng.integers(1, 5, len(sizes)).astype(float),
                      ec_transcripts=FlatECs(ptr, rng.integers(0, T, ptr[-1]).astype(np.int32)),
                      transcript_names=names, total_reads=0, n_targets=T, sample_dir="")
        libs.append(Library(f"l{l}", tcc, b=np.ones(T)))
    for level in ["gene", "transcript"]:
        em = JointEM(libs, units=[0, 1], tie=[0, 0], groups=gene, pool=level, pool_tau=1.0, threads=2)
        assert sum(len(f.tx) for f in em.libs) >= 1_000_000
        res = em.run(max_iter=6, min_iter=6)
        assert np.isfinite(res.loglik[-1])


def test_pooled_fit_with_stopping_rule_converges():
    """Accelerated pooled fits with the log-posterior stopping rule reach the unaccelerated solution
    (regression: the objective depended on stored profile state and decreases counted as convergence)."""
    from joli.joint import JointEM
    libs, gene, _ = _two_platforms(np.random.default_rng(4), G=60, gene_shift=1.0)
    for level in ["transcript", "gene"]:
        mk = lambda: JointEM(libs, units=[0, 1], tie=[0, 0], groups=gene, pool=level, pool_tau=0.5)
        ref = mk().run(rel_tol=1e-9, max_iter=50000, zero_below=None, accelerate=False)
        fast = mk().run(rel_tol=1e-2, ll_rel_tol=1e-10, max_iter=50000, zero_below=None)
        assert _close_theta(fast.theta, ref.theta, 0.01), level
        lp = np.array(ref.loglik[:-1])
        assert (np.diff(lp) > -1e-6 * abs(lp[-1])).all(), level


def test_pooled_profile_between_units():
    """The consensus profile of a tie group lies between its units (small tau: equal to them)."""
    from joli.joint import JointEM
    libs, gene, _ = _two_platforms(np.random.default_rng(5), G=60, gene_shift=1.0)
    em = JointEM(libs, units=[0, 1], tie=[7, 7], groups=gene, pool="transcript", pool_tau=0.003)
    res = em.run(rel_tol=1e-7, max_iter=50000, zero_below=None)
    labels, prof = em.pooled_profile()
    assert labels == [7] and prof.shape == (1, em.T_full)
    assert 0.5 * np.abs(prof[0] - res.theta[0]).sum() < 0.02


def _cell_types(rng, K=4, G=80, iso=3, dev=0.3, depths=None):
    """K cell types whose log abundances deviate (sd dev) from a common profile; one transcript-resolving
    library each (depths[k] reads). Returns libraries, true theta (K, T), gene index."""
    T = G * iso
    names = [f"t{i}" for i in range(T)]
    gene = np.repeat(np.arange(G), iso)
    base = rng.normal(0, 1.5, T)
    libs, thetas = [], []
    for k in range(K):
        x = base + rng.normal(0, dev, T)
        th = np.exp(x) / np.exp(x).sum()
        thetas.append(th)
        libs.append(_ec_library(f"k{k}", rng, [np.array([t]) for t in range(T)], th, depths[k], T, names, np.ones(T)))
    return libs, np.array(thetas), gene


def test_design_one_hot_flat_matches_separate_units():
    """Flat one-hot design = no sharing: equals separate unit fits when every unit sees every transcript."""
    from joli.joint import JointEM
    libs, _, gene = _cell_types(np.random.default_rng(0), depths=[200000] * 4)
    kw = dict(rel_tol=1e-8, max_iter=50000)
    sep = JointEM(libs, units=[0, 1, 2, 3]).run(**kw).theta
    X = np.eye(4)
    des = JointEM(libs, units=[0, 1, 2, 3], pool="design", design=X, design_sd=np.full(4, np.inf), pool_tau=1.0).run(**kw).theta
    assert np.all(sep > 0)
    assert _close_theta(sep, des, 0.01)


def test_design_prior_helps_sparse_cell_type():
    from joli.joint import JointEM
    libs, truth, gene = _cell_types(np.random.default_rng(1), dev=0.3, depths=[100000, 100000, 100000, 600])
    kw = dict(rel_tol=1e-7, max_iter=50000)
    sep = JointEM(libs, units=[0, 1, 2, 3]).run(**kw).theta
    X = np.column_stack([np.ones(4), np.eye(4)])                    # intercept + cell types
    des = JointEM(libs, units=[0, 1, 2, 3], pool="design", design=X,
                  design_sd=np.r_[np.inf, np.full(4, 0.3)], pool_tau=0.1).run(**kw).theta
    err = lambda th: 0.5 * np.abs(th[3] - truth[3]).sum()
    assert err(des) < 0.7 * err(sep), (err(sep), err(des))
