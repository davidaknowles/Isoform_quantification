"""
test_training_tracker.py
========================
Tests for JOLI_Kallisto/core/training_tracker.py.

Covers:
  - record() stores correct number of rounds
  - alpha_sum matches actual alpha.sum()
  - alpha_entropy is non-negative
  - alpha_max_change is None on round 0, float on round 1+
  - inter_sample_corr contains all expected pairs
  - theta_vs_alpha_corr has one entry per sample
  - nonzero_per_sample matches actual nonzero counts
  - print_round_summary() runs without error
  - save() / load() round-trip preserves history length and values

Run with:
  conda activate NanoCount_5
  cd JOLI_Kallisto/
  python test/test_training_tracker.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np


from joli.training_tracker import TrainingTracker
from test_helpers import check


# ============================================================
# Shared small-case setup
# ============================================================

SAMPLE_NAMES = ["sample_A", "sample_B", "sample_C"]
T = 10   # number of transcripts

rng = np.random.default_rng(42)

def _make_theta(rng, T):
    """Return a random normalized theta vector of length T."""
    raw = rng.dirichlet(np.ones(T))
    return raw / raw.sum()

def _make_alpha(rng, T):
    """Return a random positive alpha vector of length T."""
    return rng.uniform(0.5, 5.0, size=T)

def _record_round(tracker, gd_round, rng, T, n_samples):
    """Record one round with random data."""
    theta_list        = [_make_theta(rng, T) for _ in range(n_samples)]
    alpha             = _make_alpha(rng, T)
    em_rounds_list    = [rng.integers(10, 100) for _ in range(n_samples)]
    em_converged_list = [bool(rng.integers(0, 2)) for _ in range(n_samples)]
    gd_loss           = float(rng.uniform(1.0, 10.0))
    tracker.record(gd_round, theta_list, alpha, em_rounds_list, em_converged_list, gd_loss)
    return theta_list, alpha, em_rounds_list, em_converged_list, gd_loss


# ============================================================
# Test 1 — record() accumulates rounds
# ============================================================

def test_record_accumulates():
    """Each call to record() adds exactly one entry to history."""
    print("\n[test_record_accumulates]")
    tracker = TrainingTracker(SAMPLE_NAMES)
    passed  = True
    rng_    = np.random.default_rng(1)

    passed &= check(len(tracker.history) == 0,
                    "history is empty before any record()")

    for r in range(5):
        _record_round(tracker, r, rng_, T, len(SAMPLE_NAMES))
        passed &= check(len(tracker.history) == r + 1,
                        f"history has {r + 1} entries after round {r}")
    return passed


# ============================================================
# Test 2 — alpha_sum and alpha_entropy are correct
# ============================================================

def test_alpha_metrics():
    """alpha_sum matches alpha.sum(); alpha_entropy >= 0."""
    print("\n[test_alpha_metrics]")
    tracker = TrainingTracker(SAMPLE_NAMES)
    rng_    = np.random.default_rng(2)
    passed  = True

    theta_list    = [_make_theta(rng_, T) for _ in range(len(SAMPLE_NAMES))]
    alpha         = _make_alpha(rng_, T)
    em_rounds     = [50] * len(SAMPLE_NAMES)
    em_converged  = [True] * len(SAMPLE_NAMES)
    gd_loss       = 3.14

    tracker.record(0, theta_list, alpha, em_rounds, em_converged, gd_loss)
    rec = tracker.history[0]

    passed &= check(abs(rec["alpha_sum"] - float(alpha.sum())) < 1e-9,
                    "alpha_sum matches alpha.sum()",
                    f"got {rec['alpha_sum']:.6f} vs {alpha.sum():.6f}")
    passed &= check(rec["alpha_entropy"] >= 0,
                    "alpha_entropy >= 0",
                    f"got {rec['alpha_entropy']:.6f}")
    passed &= check(abs(rec["gd_loss"] - gd_loss) < 1e-9,
                    "gd_loss stored correctly")
    return passed


# ============================================================
# Test 3 — alpha_max_change is None on round 0, float on round 1+
# ============================================================

def test_alpha_change():
    """alpha_max_change is None for the first round, a float thereafter."""
    print("\n[test_alpha_change]")
    tracker = TrainingTracker(SAMPLE_NAMES)
    rng_    = np.random.default_rng(3)
    passed  = True

    _record_round(tracker, 0, rng_, T, len(SAMPLE_NAMES))
    passed &= check(tracker.history[0]["alpha_max_change"] is None,
                    "alpha_max_change is None on round 0")

    _record_round(tracker, 1, rng_, T, len(SAMPLE_NAMES))
    passed &= check(isinstance(tracker.history[1]["alpha_max_change"], float),
                    "alpha_max_change is float on round 1",
                    f"got {tracker.history[1]['alpha_max_change']}")
    passed &= check(tracker.history[1]["alpha_max_change"] >= 0,
                    "alpha_max_change >= 0")
    return passed


# ============================================================
# Test 4 — inter_sample_corr has all expected pairs
# ============================================================

def test_inter_sample_pairs():
    """inter_sample_corr contains all C(n,2) pairs with spearman and pearson."""
    print("\n[test_inter_sample_pairs]")
    tracker = TrainingTracker(SAMPLE_NAMES)
    rng_    = np.random.default_rng(4)
    passed  = True

    _record_round(tracker, 0, rng_, T, len(SAMPLE_NAMES))
    corr = tracker.history[0]["inter_sample_corr"]

    n       = len(SAMPLE_NAMES)
    n_pairs = n * (n - 1) // 2
    passed &= check(len(corr) == n_pairs,
                    f"inter_sample_corr has {n_pairs} pairs",
                    f"got {len(corr)}")

    for pair, vals in corr.items():
        passed &= check("spearman" in vals and "pearson" in vals,
                        f"pair {pair} has spearman and pearson keys")
        passed &= check(-1.0 <= vals["spearman"] <= 1.0,
                        f"spearman in [-1,1] for {pair}",
                        f"got {vals['spearman']:.4f}")
        passed &= check(-1.0 <= vals["pearson"] <= 1.0,
                        f"pearson in [-1,1] for {pair}",
                        f"got {vals['pearson']:.4f}")
    return passed


# ============================================================
# Test 5 — theta_vs_alpha_corr has one entry per sample
# ============================================================

def test_theta_vs_alpha_corr():
    """theta_vs_alpha_corr has len(sample_names) entries."""
    print("\n[test_theta_vs_alpha_corr]")
    tracker = TrainingTracker(SAMPLE_NAMES)
    rng_    = np.random.default_rng(5)
    passed  = True

    _record_round(tracker, 0, rng_, T, len(SAMPLE_NAMES))
    tva = tracker.history[0]["theta_vs_alpha_corr"]

    passed &= check(len(tva) == len(SAMPLE_NAMES),
                    f"theta_vs_alpha_corr length == n_samples",
                    f"got {len(tva)}")
    for i, entry in enumerate(tva):
        passed &= check("spearman" in entry and "pearson" in entry,
                        f"sample {i} has spearman and pearson keys")
    return passed


# ============================================================
# Test 6 — nonzero_per_sample is correct
# ============================================================

def test_nonzero_per_sample():
    """nonzero_per_sample matches actual (theta > 0).sum() per sample."""
    print("\n[test_nonzero_per_sample]")
    tracker = TrainingTracker(["s1", "s2"])
    passed  = True

    # Construct theta with known nonzero counts
    theta_a = np.array([0.5, 0.0, 0.3, 0.2, 0.0])  # 3 nonzero
    theta_b = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # 5 nonzero
    alpha   = np.ones(5)

    tracker.record(0, [theta_a, theta_b], alpha, [10, 20], [True, True], 1.0)
    nz = tracker.history[0]["nonzero_per_sample"]

    passed &= check(nz[0] == 3,
                    "sample_a nonzero == 3",
                    f"got {nz[0]}")
    passed &= check(nz[1] == 5,
                    "sample_b nonzero == 5",
                    f"got {nz[1]}")
    return passed


# ============================================================
# Test 7 — print_round_summary does not crash
# ============================================================

def test_print_round_summary():
    """print_round_summary() runs without error after one record()."""
    print("\n[test_print_round_summary]")
    tracker = TrainingTracker(SAMPLE_NAMES)
    rng_    = np.random.default_rng(6)
    passed  = True

    _record_round(tracker, 0, rng_, T, len(SAMPLE_NAMES))

    no_error = True
    try:
        tracker.print_round_summary(0)
    except Exception as e:
        no_error = False
        print(f"    Error: {e}")
    passed &= check(no_error, "print_round_summary() runs without error")
    return passed


# ============================================================
# Test 8 — save / load round-trip
# ============================================================

def test_save_load_roundtrip():
    """save() then load() preserves history length and key values."""
    print("\n[test_save_load_roundtrip]")
    tracker = TrainingTracker(SAMPLE_NAMES)
    rng_    = np.random.default_rng(7)
    passed  = True

    for r in range(3):
        _record_round(tracker, r, rng_, T, len(SAMPLE_NAMES))

    tmpdir = tempfile.mkdtemp(prefix="joli_tracker_test_")
    try:
        path = os.path.join(tmpdir, "tracker.pkl")
        tracker.save(path)

        loaded = TrainingTracker.load(path)
        passed &= check(len(loaded.history) == 3,
                        "loaded tracker has 3 rounds",
                        f"got {len(loaded.history)}")
        passed &= check(loaded.sample_names == SAMPLE_NAMES,
                        "sample_names preserved after load")
        passed &= check(
            abs(loaded.history[0]["gd_loss"] - tracker.history[0]["gd_loss"]) < 1e-9,
            "gd_loss preserved after load"
        )
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Runner
# ============================================================

def run_all():
    tests = [
        test_record_accumulates,
        test_alpha_metrics,
        test_alpha_change,
        test_inter_sample_pairs,
        test_theta_vs_alpha_corr,
        test_nonzero_per_sample,
        test_print_round_summary,
        test_save_load_roundtrip,
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
    import sys as _sys
    success = run_all()
    _sys.exit(0 if success else 1)
