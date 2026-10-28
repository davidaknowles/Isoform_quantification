"""
test_em_algorithm.py
====================
Tests for JOLI_Kallisto/em_algorithm.py.

Covers:
  - Plain EM (alpha_prior=None): backward-compatibility check
  - MAP EM (alpha_prior provided): posterior-mean M-step formula
  - init_theta warm-start: EM starts from given theta, not uniform
  - alpha_prior shape validation: ValueError on wrong shape
  - Uniform prior (alpha=1) matches plain EM closely
  - Strong asymmetric prior pulls theta toward prior direction
  - init_theta=zeros falls back to uniform gracefully

Run with:
  conda activate NanoCount_5
  cd JOLI_Kallisto/
  python test/test_em_algorithm.py

All tests print PASS / FAIL. No external test framework required.
"""

import os
import shutil
import sys

import numpy as np

# Add JOLI_Kallisto directory to path for module imports

from joli.load_tcc import load_tcc_data
from joli.weights import compute_weights
from joli.em_algorithm import JoliEM, EMResult
from test_helpers import make_sample_dir, check


# ============================================================
# Shared small-case setup
#
# Layout:
#   3 transcripts: tx0, tx1, tx2
#   EC0: [tx0]        count=50  (single-tx → added post-convergence)
#   EC1: [tx1]        count=30  (single-tx → added post-convergence)
#   EC2: [tx0, tx1]   count=40  (multi-tx  → EM assigns fractionally)
#   EC3: [tx1, tx2]   count=20  (multi-tx  → EM assigns fractionally)
# ============================================================

EC_TRANSCRIPTS   = [[0], [1], [0, 1], [1, 2]]
EC_COUNTS        = [50, 30, 40, 20]
TRANSCRIPT_NAMES = ["ENST_tx0", "ENST_tx1", "ENST_tx2"]


def make_em(tmpdir):
    """Load TCC data and return a JoliEM instance for the shared test case."""
    tcc_data    = load_tcc_data(tmpdir)
    weight_data = compute_weights(tcc_data, mode="uniform")
    return JoliEM(tcc_data, weight_data)


# ============================================================
# Test 1 — Plain EM unchanged (backward compatibility)
# ============================================================

def test_plain_em_backward_compat():
    """
    alpha_prior=None must give same result as before the MAP changes.
    Checks: EMResult returned, alpha shape correct, nonzero transcripts > 0.
    """
    print("\n[test_plain_em_backward_compat]")
    tmpdir = make_sample_dir(EC_TRANSCRIPTS, EC_COUNTS, TRANSCRIPT_NAMES)
    passed = True
    try:
        em     = make_em(tmpdir)
        result = em.run(convergence_mode="joli", alpha_prior=None)

        passed &= check(isinstance(result, EMResult),
                        "run() returns EMResult")
        passed &= check(result.theta_unnorm.shape == (3,),
                        "alpha shape == (3,)",
                        f"got {result.theta_unnorm.shape}")
        passed &= check(result.theta_unnorm.sum() > 0,
                        "alpha sum > 0")
        passed &= check(int((result.theta_unnorm > 0).sum()) > 0,
                        "at least one nonzero transcript")
        # Single-tx counts must appear in alpha (Fix A)
        passed &= check(result.theta_unnorm[0] >= 50,
                        "tx0 alpha >= 50 (includes single-tx EC0 count)",
                        f"got {result.theta_unnorm[0]:.4f}")
        passed &= check(result.theta_unnorm[1] >= 30,
                        "tx1 alpha >= 30 (includes single-tx EC1 count)",
                        f"got {result.theta_unnorm[1]:.4f}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Test 2 — alpha_prior shape validation
# ============================================================

def test_alpha_prior_shape_validation():
    """
    Passing an alpha_prior with wrong shape must raise ValueError.
    """
    print("\n[test_alpha_prior_shape_validation]")
    tmpdir = make_sample_dir(EC_TRANSCRIPTS, EC_COUNTS, TRANSCRIPT_NAMES)
    passed = True
    try:
        em = make_em(tmpdir)

        # Wrong shape: 2 instead of 3
        raised = False
        try:
            em.run(convergence_mode="joli",
                   alpha_prior=np.ones(2, dtype=np.float64))
        except ValueError:
            raised = True
        passed &= check(raised,
                        "ValueError raised for alpha_prior shape (2,) vs n_transcripts=3")

        # Correct shape: no error
        no_error = True
        try:
            em.run(convergence_mode="joli",
                   alpha_prior=np.ones(3, dtype=np.float64))
        except Exception:
            no_error = False
        passed &= check(no_error,
                        "No error for alpha_prior shape (3,) — correct shape")
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Test 3 — Uniform prior (alpha=1) close to plain EM
# ============================================================

def test_uniform_prior_matches_plain_em():
    """
    alpha_prior = ones(T) (uniform Dirichlet) adds equal pseudocounts to all
    transcripts. With large read counts the prior has negligible effect, so
    MAP and plain EM results should be very close (Pearson > 0.9999).
    """
    print("\n[test_uniform_prior_matches_plain_em]")
    # Use larger counts so the prior (alpha=1 per transcript) is negligible
    ec_transcripts   = [[0], [1], [0, 1], [1, 2]]
    ec_counts        = [5000, 3000, 4000, 2000]
    transcript_names = ["tx0", "tx1", "tx2"]

    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names)
    passed = True
    try:
        tcc_data    = load_tcc_data(tmpdir)
        weight_data = compute_weights(tcc_data, mode="uniform")
        em          = JoliEM(tcc_data, weight_data)

        result_plain = em.run(convergence_mode="joli", alpha_prior=None)
        result_map   = em.run(convergence_mode="joli",
                              alpha_prior=np.ones(3, dtype=np.float64))

        # Normalize to compare proportions
        plain_norm = result_plain.theta_unnorm / result_plain.theta_unnorm.sum()
        map_norm   = result_map.theta_unnorm   / result_map.theta_unnorm.sum()

        corr = float(np.corrcoef(plain_norm, map_norm)[0, 1])
        passed &= check(corr > 0.9999,
                        f"Pearson(plain, MAP uniform prior) > 0.9999",
                        f"got {corr:.6f}")

        max_diff = float(np.abs(plain_norm - map_norm).max())
        passed &= check(max_diff < 0.01,
                        "max absolute diff of normalized alpha < 0.01",
                        f"got {max_diff:.6f}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Test 4 — Strong asymmetric prior pulls theta toward prior
# ============================================================

def test_strong_prior_pulls_theta():
    """
    With a strong prior heavily favoring tx2 (alpha[2] >> alpha[0], alpha[1]),
    MAP theta[2] should be substantially larger than plain EM theta[2].

    Uses ONLY multi-tx ECs so single-tx counts don't dilute the prior effect.
    EC0: [tx0, tx1, tx2]  count=100  (all reads compete across all 3 transcripts)
    Plain EM: uniform theta → tx2 gets ~1/3.
    MAP with alpha=[1,1,1000]: tx2 should dominate (fraction >> 0.5).
    """
    print("\n[test_strong_prior_pulls_theta]")
    # Pure multi-tx case: no single-tx ECs so prior effect is unambiguous
    ec_transcripts   = [[0, 1, 2]]
    ec_counts        = [100]
    transcript_names = ["tx0", "tx1", "tx2"]

    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names)
    passed = True
    try:
        tcc_data    = load_tcc_data(tmpdir)
        weight_data = compute_weights(tcc_data, mode="uniform")
        em          = JoliEM(tcc_data, weight_data)

        result_plain = em.run(convergence_mode="joli", alpha_prior=None)

        # Strong prior: tx2 gets 1000x more weight than tx0 and tx1
        alpha_prior  = np.array([1.0, 1.0, 1000.0], dtype=np.float64)
        result_map   = em.run(convergence_mode="joli", alpha_prior=alpha_prior)

        plain_tx2 = result_plain.theta_unnorm[2] / result_plain.theta_unnorm.sum()
        map_tx2   = result_map.theta_unnorm[2]   / result_map.theta_unnorm.sum()

        passed &= check(map_tx2 > plain_tx2,
                        "Strong prior on tx2: MAP tx2 fraction > plain EM tx2 fraction",
                        f"plain={plain_tx2:.4f}, map={map_tx2:.4f}")
        passed &= check(map_tx2 > 0.5,
                        "Strong prior on tx2: MAP tx2 fraction > 0.5",
                        f"got {map_tx2:.4f}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Test 5 — init_theta warm-start is used
# ============================================================

def test_init_theta_warm_start():
    """
    When init_theta is provided, EM must start from that theta instead of
    uniform. We verify by starting from a theta heavily biased toward tx0
    and checking the EM converges to a different result than from uniform
    (or at minimum: runs without error and returns EMResult).

    Also verifies init_theta=zeros falls back to uniform gracefully.
    """
    print("\n[test_init_theta_warm_start]")
    tmpdir = make_sample_dir(EC_TRANSCRIPTS, EC_COUNTS, TRANSCRIPT_NAMES)
    passed = True
    try:
        tcc_data    = load_tcc_data(tmpdir)
        weight_data = compute_weights(tcc_data, mode="uniform")
        em          = JoliEM(tcc_data, weight_data)

        # Warm-start from a valid theta
        init = np.array([0.8, 0.1, 0.1], dtype=np.float64)
        result = em.run(convergence_mode="joli",
                        alpha_prior=None,
                        init_theta=init)
        passed &= check(isinstance(result, EMResult),
                        "run() with init_theta returns EMResult")
        passed &= check(result.theta_unnorm.shape == (3,),
                        "alpha shape correct with init_theta")

        # Warm-start from zeros must not crash — falls back to uniform
        result_zeros = em.run(convergence_mode="joli",
                              alpha_prior=None,
                              init_theta=np.zeros(3, dtype=np.float64))
        passed &= check(isinstance(result_zeros, EMResult),
                        "init_theta=zeros falls back gracefully, returns EMResult")
        passed &= check(result_zeros.theta_unnorm.sum() > 0,
                        "init_theta=zeros: alpha sum > 0 after fallback")
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Test 6 — Posterior-mean formula: one-round manual verification
# ============================================================

def test_posterior_mean_formula_one_round():
    """
    Manually compute the expected theta after ONE MAP EM round and compare
    with what JoliEM produces.

    Setup: 2 transcripts, 1 multi-tx EC only (no single-tx).
      EC0: [tx0, tx1]  count=100
      Uniform weights (eff_len=1 for both)

    Round 0 theta (uniform): [0.5, 0.5]
    E-step:
      denom_EC0 = 0.5 * 1 + 0.5 * 1 = 1.0
      n[tx0] = 100 * 0.5 / 1.0 = 50
      n[tx1] = 100 * 0.5 / 1.0 = 50

    MAP M-step with alpha_prior = [10, 2]:
      numerator = [50+10, 50+2] = [60, 52]
      theta_new = [60/112, 52/112] = [0.5357, 0.4643]
    """
    print("\n[test_posterior_mean_formula_one_round]")
    ec_transcripts   = [[0, 1]]
    ec_counts        = [100]
    transcript_names = ["tx0", "tx1"]

    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names)
    passed = True
    try:
        tcc_data    = load_tcc_data(tmpdir)
        weight_data = compute_weights(tcc_data, mode="uniform")
        em          = JoliEM(tcc_data, weight_data)

        alpha_prior = np.array([10.0, 2.0], dtype=np.float64)

        # Run exactly 1 round (min_rounds=1, max_em_rounds=1)
        result = em.run(
            max_em_rounds=1,
            min_rounds=1,
            convergence_mode="joli",
            alpha_prior=alpha_prior,
        )

        # Expected after 1 MAP round:
        # n = [50, 50] (from E-step with uniform init)
        # numerator = [60, 52], sum = 112
        # theta = [60/112, 52/112] — no single-tx ECs so alpha = theta * 100
        expected_alpha = np.array([60.0 / 112.0 * 100, 52.0 / 112.0 * 100])

        diff = np.abs(result.theta_unnorm - expected_alpha)
        passed &= check(diff.max() < 1e-6,
                        "MAP alpha after 1 round matches manual formula",
                        f"expected={expected_alpha}, got={result.theta_unnorm}, diff={diff}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Runner
# ============================================================

def run_all():
    tests = [
        test_plain_em_backward_compat,
        test_alpha_prior_shape_validation,
        test_uniform_prior_matches_plain_em,
        test_strong_prior_pulls_theta,
        test_init_theta_warm_start,
        test_posterior_mean_formula_one_round,
    ]

    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception as e:
            print(f"  ERROR in {t.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    passed = sum(results)
    total  = len(results)
    print(f"Results: {passed}/{total} test functions passed")
    if passed == total:
        print("ALL PASSED")
    else:
        print("SOME FAILED — see details above")
    return passed == total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
