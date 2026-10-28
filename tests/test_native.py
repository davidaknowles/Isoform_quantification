"""The compiled E-step kernels match the numpy reference."""

import numpy as np
import pytest

from joli import native
from joli.joint import JointEM, Library
from joli.load_tcc import TCCData

pytestmark = pytest.mark.skipif(not native.available(), reason="rust/joli_kernels not built")


def random_library(rng, T=500, E=3000):
    ecs = [np.sort(rng.choice(T, size=rng.integers(1, 6), replace=False)) for _ in range(E)]
    counts = rng.poisson(5, size=E).astype(float)
    c = [rng.gamma(2.0, size=len(e)) for e in ecs]
    tcc = TCCData(ec_counts=counts, ec_transcripts=ecs, transcript_names=[f"t{i}" for i in range(T)],
                  total_reads=int(counts.sum()), n_targets=T, sample_dir="")
    return Library("x", tcc, b=rng.uniform(100, 2000, size=T), c=c)


def test_rust_matches_numpy():
    rng = np.random.default_rng(0)
    libs = [random_library(rng), random_library(rng)]
    fits = {}
    for backend in ["numpy", "rust"]:
        for threads in ([1, 4] if backend == "rust" else [None]):
            em = JointEM(libs, units=[0, 1], backend=backend, threads=threads)
            theta = rng.dirichlet(np.ones(em.T), size=2) if backend == "numpy" else fits["theta0"]
            if backend == "numpy":
                fits["theta0"] = theta
            Z, D, ll, R = em._estep(theta)
            z = em.position_counts(theta, 1)
            fits[(backend, threads)] = (Z, ll, z, em.run(max_iter=200).theta)
    ref = fits[("numpy", None)]
    for key in [("rust", 1), ("rust", 4)]:
        Z, ll, z, th = fits[key]
        np.testing.assert_allclose(Z, ref[0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ll, ref[1], rtol=1e-12)
        np.testing.assert_allclose(z, ref[2], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(th, ref[3], rtol=1e-8, atol=1e-12)


def test_multithreaded_kernels_match_numpy():
    rng = np.random.default_rng(1)
    lib = random_library(rng, T=20000, E=150000)          # ~450k positions: 4 threads used
    em_np = JointEM([lib], backend="numpy")
    em_rs = JointEM([lib], backend="rust", threads=4)
    assert em_rs.libs[0].ec_ptr[-1] > 4 * 100_000
    theta = rng.dirichlet(np.ones(em_np.T))[None, :]
    Z0, _, ll0, _ = em_np._estep(theta)
    Z1, _, ll1, _ = em_rs._estep(theta)
    np.testing.assert_allclose(Z1, Z0, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ll1, ll0, rtol=1e-12)
    np.testing.assert_allclose(em_rs.position_counts(theta, 0), em_np.position_counts(theta, 0), rtol=1e-10, atol=1e-12)
