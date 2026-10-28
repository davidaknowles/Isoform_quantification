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
