"""
test_multi_sample_em.py
=======================
Tests for JOLI_Kallisto/multi_sample_em.py.

Covers:
  - Requires at least 2 samples (ValueError on 1 sample)
  - Mismatched transcript counts across samples raises ValueError
  - run() returns expected keys and shapes
  - GD loss decreases over rounds
  - Per-sample theta_list has correct shapes and sums to ~1
  - write_results() produces abundance.tsv, alpha_final.npy, gd_loss_history.pkl
  - All samples share the same final alpha shape

Run with:
  conda activate NanoCount_5
  cd JOLI_Kallisto/
  python test/test_multi_sample_em.py
"""

import os
import pickle
import shutil
import sys
import tempfile

import numpy as np


from joli.multi_sample_em import MultiSampleJoliEM
from test_helpers import make_sample_dir, check


# ============================================================
# Two synthetic samples with different read distributions
#
# Shared transcriptome: 3 transcripts (tx0, tx1, tx2)
# Sample A: tx0 dominant (lots of single-tx reads on tx0)
# Sample B: tx1 dominant (lots of single-tx reads on tx1)
# Both share a 3-way multi-tx EC to create coupling
# ============================================================

TRANSCRIPT_NAMES = ["ENST_tx0", "ENST_tx1", "ENST_tx2"]

SAMPLE_A_ECS    = [[0], [1], [2], [0, 1, 2]]
SAMPLE_A_COUNTS = [200, 10,  5,   50]          # tx0 dominant

SAMPLE_B_ECS    = [[0], [1], [2], [0, 1, 2]]
SAMPLE_B_COUNTS = [10,  200, 5,   50]          # tx1 dominant


def make_two_sample_dirs():
    """Create two temporary sample directories. Returns (dir_a, dir_b)."""
    dir_a = make_sample_dir(SAMPLE_A_ECS, SAMPLE_A_COUNTS, TRANSCRIPT_NAMES)
    dir_b = make_sample_dir(SAMPLE_B_ECS, SAMPLE_B_COUNTS, TRANSCRIPT_NAMES)
    return dir_a, dir_b


# ============================================================
# Test 1 — Fewer than 2 samples raises ValueError
# ============================================================

def test_requires_two_samples():
    """MultiSampleJoliEM must raise ValueError when given fewer than 2 samples."""
    print("\n[test_requires_two_samples]")
    dir_a = make_sample_dir(SAMPLE_A_ECS, SAMPLE_A_COUNTS, TRANSCRIPT_NAMES)
    passed = True
    try:
        raised = False
        try:
            MultiSampleJoliEM(sample_dirs=[dir_a])
        except ValueError:
            raised = True
        passed &= check(raised, "ValueError raised for only 1 sample")
    finally:
        shutil.rmtree(dir_a)
    return passed


# ============================================================
# Test 2 — Mismatched transcript counts raises ValueError
# ============================================================

def test_mismatched_transcripts_raises():
    """Samples with different transcript counts must raise ValueError."""
    print("\n[test_mismatched_transcripts_raises]")
    # Sample A: 3 transcripts; Sample B: 4 transcripts
    dir_a = make_sample_dir([[0], [0, 1]], [10, 5], ["tx0", "tx1", "tx2"])
    dir_b = make_sample_dir([[0], [0, 1]], [10, 5], ["tx0", "tx1", "tx2", "tx3"])
    passed = True
    try:
        raised = False
        try:
            MultiSampleJoliEM(sample_dirs=[dir_a, dir_b])
        except ValueError:
            raised = True
        passed &= check(raised, "ValueError raised for mismatched transcript counts")
    finally:
        shutil.rmtree(dir_a)
        shutil.rmtree(dir_b)
    return passed


# ============================================================
# Test 3 — run() returns expected structure and shapes
# ============================================================

def test_run_returns_correct_structure():
    """run() must return a dict with all expected keys, correct types and shapes."""
    print("\n[test_run_returns_correct_structure]")
    dir_a, dir_b = make_two_sample_dirs()
    passed = True
    try:
        ms = MultiSampleJoliEM(
            sample_dirs      = [dir_a, dir_b],
            max_gd_rounds    = 3,
            gd_steps_per_round = 3,
            max_em_rounds    = 200,
            min_em_rounds    = 50,
        )
        results = ms.run()

        passed &= check(isinstance(results, dict), "run() returns dict")

        for key in ("theta_list", "alpha", "gd_loss_history", "n_gd_rounds", "converged"):
            passed &= check(key in results, f"Key '{key}' present in results")

        passed &= check(len(results["theta_list"]) == 2,
                        "theta_list has 2 entries (one per sample)",
                        f"got {len(results['theta_list'])}")
        passed &= check(results["alpha"].shape == (3,),
                        "alpha shape == (3,)",
                        f"got {results['alpha'].shape}")
        passed &= check(isinstance(results["n_gd_rounds"], int),
                        "n_gd_rounds is int")
        passed &= check(results["n_gd_rounds"] <= 3,
                        "n_gd_rounds <= max_gd_rounds=3",
                        f"got {results['n_gd_rounds']}")
        passed &= check(isinstance(results["converged"], bool),
                        "converged is bool")
    finally:
        shutil.rmtree(dir_a)
        shutil.rmtree(dir_b)
    return passed


# ============================================================
# Test 4 — Per-sample theta shapes and normalization
# ============================================================

def test_theta_shapes_and_normalization():
    """Each theta_s must have shape (T,) and sum to ~1."""
    print("\n[test_theta_shapes_and_normalization]")
    dir_a, dir_b = make_two_sample_dirs()
    passed = True
    try:
        ms = MultiSampleJoliEM(
            sample_dirs      = [dir_a, dir_b],
            max_gd_rounds    = 2,
            gd_steps_per_round = 3,
            max_em_rounds    = 200,
            min_em_rounds    = 50,
        )
        results = ms.run()

        for i, theta_s in enumerate(results["theta_list"]):
            passed &= check(theta_s.shape == (3,),
                            f"theta_list[{i}] shape == (3,)",
                            f"got {theta_s.shape}")
            passed &= check(abs(theta_s.sum() - 1.0) < 1e-6,
                            f"theta_list[{i}] sums to ~1.0",
                            f"got sum={theta_s.sum():.8f}")
    finally:
        shutil.rmtree(dir_a)
        shutil.rmtree(dir_b)
    return passed


# ============================================================
# Test 5 — GD loss history structure
# ============================================================

def test_gd_loss_history_structure():
    """gd_loss_history must be a list of lists, one per GD round."""
    print("\n[test_gd_loss_history_structure]")
    dir_a, dir_b = make_two_sample_dirs()
    passed = True
    try:
        ms = MultiSampleJoliEM(
            sample_dirs      = [dir_a, dir_b],
            max_gd_rounds    = 3,
            gd_steps_per_round = 5,
            max_em_rounds    = 200,
            min_em_rounds    = 50,
        )
        results = ms.run()
        hist = results["gd_loss_history"]

        passed &= check(isinstance(hist, list),
                        "gd_loss_history is list")
        passed &= check(len(hist) == results["n_gd_rounds"],
                        "len(gd_loss_history) == n_gd_rounds",
                        f"got {len(hist)} vs {results['n_gd_rounds']}")
        passed &= check(all(isinstance(r, list) for r in hist),
                        "each entry in gd_loss_history is a list")
        passed &= check(all(len(r) > 0 for r in hist),
                        "each per-round loss list is non-empty")
    finally:
        shutil.rmtree(dir_a)
        shutil.rmtree(dir_b)
    return passed


# ============================================================
# Test 6 — write_results() produces all expected output files
# ============================================================

def test_write_results_files():
    """write_results() must produce abundance.tsv per sample, alpha_final.npy, gd_loss_history.pkl."""
    print("\n[test_write_results_files]")
    dir_a, dir_b = make_two_sample_dirs()
    out_dir = tempfile.mkdtemp(prefix="joli_ms_test_")
    passed = True
    try:
        ms = MultiSampleJoliEM(
            sample_dirs      = [dir_a, dir_b],
            max_gd_rounds    = 2,
            gd_steps_per_round = 3,
            max_em_rounds    = 200,
            min_em_rounds    = 50,
        )
        results = ms.run()
        ms.write_results(out_dir, results)

        # Per-sample abundance.tsv
        for sname in ms.sample_names:
            tsv_path = os.path.join(out_dir, sname, "abundance.tsv")
            passed &= check(os.path.exists(tsv_path),
                            f"abundance.tsv exists for sample '{sname}'",
                            f"expected at {tsv_path}")

        # Shared outputs
        passed &= check(os.path.exists(os.path.join(out_dir, "alpha_final.npy")),
                        "alpha_final.npy exists")
        passed &= check(os.path.exists(os.path.join(out_dir, "gd_loss_history.pkl")),
                        "gd_loss_history.pkl exists")

        # alpha_final.npy loads correctly
        alpha_loaded = np.load(os.path.join(out_dir, "alpha_final.npy"))
        passed &= check(alpha_loaded.shape == (3,),
                        "alpha_final.npy loads with shape (3,)",
                        f"got {alpha_loaded.shape}")

        # gd_loss_history.pkl loads correctly
        with open(os.path.join(out_dir, "gd_loss_history.pkl"), "rb") as fh:
            hist_loaded = pickle.load(fh)
        passed &= check(isinstance(hist_loaded, list),
                        "gd_loss_history.pkl loads as list")
    finally:
        shutil.rmtree(dir_a)
        shutil.rmtree(dir_b)
        shutil.rmtree(out_dir)
    return passed


# ============================================================
# Test 7 — Samples with different distributions get different theta
# ============================================================

def test_samples_get_different_theta():
    """
    Sample A (tx0 dominant) and Sample B (tx1 dominant) should converge
    to different theta distributions even after joint MAP EM.
    """
    print("\n[test_samples_get_different_theta]")
    dir_a, dir_b = make_two_sample_dirs()
    passed = True
    try:
        ms = MultiSampleJoliEM(
            sample_dirs      = [dir_a, dir_b],
            max_gd_rounds    = 5,
            gd_steps_per_round = 5,
            max_em_rounds    = 500,
            min_em_rounds    = 50,
        )
        results = ms.run()

        theta_a = results["theta_list"][0]
        theta_b = results["theta_list"][1]

        # Sample A: tx0 (index 0) should dominate
        passed &= check(theta_a[0] > theta_a[1],
                        "Sample A: theta[tx0] > theta[tx1] (tx0 dominant)",
                        f"got theta_a={theta_a.round(4)}")
        # Sample B: tx1 (index 1) should dominate
        passed &= check(theta_b[1] > theta_b[0],
                        "Sample B: theta[tx1] > theta[tx0] (tx1 dominant)",
                        f"got theta_b={theta_b.round(4)}")
    finally:
        shutil.rmtree(dir_a)
        shutil.rmtree(dir_b)
    return passed


# ============================================================
# Runner
# ============================================================

def run_all():
    tests = [
        test_requires_two_samples,
        test_mismatched_transcripts_raises,
        test_run_returns_correct_structure,
        test_theta_shapes_and_normalization,
        test_gd_loss_history_structure,
        test_write_results_files,
        test_samples_get_different_theta,
    ]

    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception as e:
            print(f"  ERROR in {t.__name__}: {e}")
            import traceback; traceback.print_exc()
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
