"""
test_dirichlet_optimizer.py
===========================
Tests for JOLI_Kallisto/dirichlet_optimizer.py.

Covers:
  - Initialization: alpha starts at alpha_initial for all transcripts
  - get_alpha(): returns correct shape and values
  - update(): returns alpha (T,) and loss_history list
  - Loss decreases after update (GD is working)
  - Wrong theta_matrix shape raises ValueError
  - Zero theta values are handled (floored, no crash)
  - Strong signal: alpha converges toward data (high-signal theta pushed up)
  - Alpha stays positive throughout (exp parameterization)

Run with:
  conda activate NanoCount_5
  cd JOLI_Kallisto/
  python test/test_dirichlet_optimizer.py
"""

import os
import sys

import numpy as np


from joli.dirichlet_optimizer import DirichletOptimizer
from test_helpers import check


# ============================================================
# Test 1 — Initialization
# ============================================================

def test_initialization():
    """Alpha is initialized to alpha_initial for all transcripts."""
    print("\n[test_initialization]")
    passed = True

    opt = DirichletOptimizer(n_transcripts=5, gd_lr=0.01, alpha_initial=2.0)
    alpha = opt.get_alpha()

    passed &= check(alpha.shape == (5,),
                    "get_alpha() shape == (5,)",
                    f"got {alpha.shape}")
    passed &= check(np.allclose(alpha, 2.0, atol=1e-5),
                    "all alpha values initialized to 2.0",
                    f"got {alpha}")
    passed &= check(alpha.dtype == np.float64,
                    "alpha dtype is float64",
                    f"got {alpha.dtype}")
    return passed


# ============================================================
# Test 2 — update() returns correct types and shapes
# ============================================================

def test_update_return_types():
    """update() returns (alpha array of shape (T,), loss_history list)."""
    print("\n[test_update_return_types]")
    passed = True

    T = 4
    S = 3
    opt = DirichletOptimizer(n_transcripts=T, gd_lr=0.01)

    # Uniform theta across samples
    theta_matrix = np.full((S, T), 1.0 / T, dtype=np.float64)
    alpha, loss_history = opt.update(theta_matrix, max_iterations=5)

    passed &= check(isinstance(alpha, np.ndarray),
                    "alpha is np.ndarray")
    passed &= check(alpha.shape == (T,),
                    f"alpha shape == ({T},)",
                    f"got {alpha.shape}")
    passed &= check(isinstance(loss_history, list),
                    "loss_history is list")
    passed &= check(len(loss_history) <= 5,
                    "loss_history length <= max_iterations=5",
                    f"got {len(loss_history)}")
    passed &= check(all(isinstance(v, float) for v in loss_history),
                    "all loss_history values are float")
    return passed


# ============================================================
# Test 3 — Loss decreases after update
# ============================================================

def test_loss_decreases():
    """
    GD loss should decrease (or stay flat) over iterations.
    Uses a non-trivial theta_matrix so alpha has a clear direction to move.
    """
    print("\n[test_loss_decreases]")
    passed = True

    T = 10
    S = 4
    rng = np.random.default_rng(42)

    # Different theta per sample (non-uniform signal)
    raw = rng.dirichlet(alpha=np.ones(T) * 0.5, size=S)   # shape (S, T)
    theta_matrix = raw / raw.sum(axis=1, keepdims=True)

    opt = DirichletOptimizer(n_transcripts=T, gd_lr=0.01)
    _, loss_history = opt.update(theta_matrix, max_iterations=20)

    # Loss at end should be <= loss at start
    passed &= check(loss_history[-1] <= loss_history[0] + 1e-3,
                    "Final loss <= initial loss (GD is reducing loss)",
                    f"start={loss_history[0]:.4f}, end={loss_history[-1]:.4f}")
    return passed


# ============================================================
# Test 4 — Wrong theta_matrix shape raises ValueError
# ============================================================

def test_wrong_shape_raises():
    """update() must raise ValueError when theta_matrix columns != n_transcripts."""
    print("\n[test_wrong_shape_raises]")
    passed = True

    opt = DirichletOptimizer(n_transcripts=5, gd_lr=0.01)

    # Wrong number of columns
    raised = False
    try:
        opt.update(np.ones((3, 7)) / 7.0)  # 7 columns, expects 5
    except ValueError:
        raised = True
    passed &= check(raised, "ValueError raised for wrong theta_matrix columns (7 vs 5)")

    # Correct shape: no error
    no_error = True
    try:
        opt.update(np.full((3, 5), 0.2))   # 5 columns, correct
    except Exception:
        no_error = False
    passed &= check(no_error, "No error for correct theta_matrix shape (3, 5)")

    return passed


# ============================================================
# Test 5 — Zero theta values handled gracefully
# ============================================================

def test_zero_theta_no_crash():
    """
    theta_matrix rows with zero entries must not crash (floored to THETA_FLOOR).
    """
    print("\n[test_zero_theta_no_crash]")
    passed = True

    T = 4
    # Row 0: some zeros; row 1: all equal
    theta_matrix = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ], dtype=np.float64)

    opt = DirichletOptimizer(n_transcripts=T, gd_lr=0.01)
    no_error = True
    try:
        alpha, _ = opt.update(theta_matrix, max_iterations=5)
    except Exception as e:
        no_error = False
        print(f"    Error: {e}")

    passed &= check(no_error, "No crash with zero theta entries")
    if no_error:
        passed &= check((alpha > 0).all(),
                        "All alpha values remain positive after update",
                        f"got min={alpha.min():.6f}")
    return passed


# ============================================================
# Test 6 — Alpha stays positive (exp parameterization)
# ============================================================

def test_alpha_always_positive():
    """
    After many GD iterations alpha must stay strictly positive (> 0).
    """
    print("\n[test_alpha_always_positive]")
    passed = True

    T = 20
    S = 5
    rng = np.random.default_rng(7)
    raw = rng.dirichlet(np.ones(T) * 0.1, size=S)   # sparse, some near-zero
    theta_matrix = raw / raw.sum(axis=1, keepdims=True)

    opt = DirichletOptimizer(n_transcripts=T, gd_lr=0.05)
    alpha, _ = opt.update(theta_matrix, max_iterations=50)

    passed &= check((alpha > 0).all(),
                    "All alpha > 0 after 50 GD iterations",
                    f"min alpha = {alpha.min():.2e}")
    return passed


# ============================================================
# Test 7 — Strong signal: alpha moves toward data
# ============================================================

def test_alpha_moves_toward_data():
    """
    With all samples concentrating on tx0, after GD alpha[0] should be
    the largest component (GD found that tx0 is consistently expressed).
    """
    print("\n[test_alpha_moves_toward_data]")
    passed = True

    T  = 5
    S  = 10
    # All samples heavily weight tx0
    theta_matrix        = np.full((S, T), 0.01, dtype=np.float64)
    theta_matrix[:, 0]  = 0.96   # tx0 gets 96% in every sample
    theta_matrix       /= theta_matrix.sum(axis=1, keepdims=True)

    opt = DirichletOptimizer(n_transcripts=T, gd_lr=0.05)
    alpha, _ = opt.update(theta_matrix, max_iterations=100)

    passed &= check(alpha[0] == alpha.max(),
                    "alpha[0] is the largest after GD on tx0-dominant data",
                    f"alpha={alpha.round(4)}")
    passed &= check(alpha[0] > alpha[1:].mean() * 2,
                    "alpha[0] > 2x mean of other alphas",
                    f"alpha[0]={alpha[0]:.4f}, mean others={alpha[1:].mean():.4f}")
    return passed


# ============================================================
# Runner
# ============================================================

def run_all():
    tests = [
        test_initialization,
        test_update_return_types,
        test_loss_decreases,
        test_wrong_shape_raises,
        test_zero_theta_no_crash,
        test_alpha_always_positive,
        test_alpha_moves_toward_data,
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
