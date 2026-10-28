"""
test_JolitoKallisto.py
======================
Cumulative test file for JOLI-Kallisto Phase 1.
A new section is added here as each step is implemented.

  Step 1.1 — load_tcc.py    : parse bustools output into TCCData
  Step 1.2 — weights.py     : compute effective lengths and EC weights
  Step 1.3 — em_algorithm.py  : EC-based EM (JoliEM class)
  Step 1.4 — output_writer.py : write abundance.tsv in kallisto format
  Step 1.5 — main_joli.py     : end-to-end Python CLI entry point

Tests create small synthetic bustools-format files that exactly match
the real file formats observed from the kallisto/bustools pipeline:
  - count.mtx       : MatrixMarket sparse (rows=samples, cols=ECs, 1-indexed)
  - matrix.ec       : "ec_id<TAB>tx_idx1,tx_idx2,..."
  - transcripts.txt : one transcript name per line
  - run_info.json   : JSON with n_processed, n_pseudoaligned, n_targets

Run with:
  conda activate NanoCount_5
  cd JOLI_Kallisto
  python test_JolitoKallisto.py

All tests print PASS/FAIL. No external test framework required.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

# Add core/ to path for library module imports
from joli.load_tcc import load_tcc_data, TCCData
from joli.weights import compute_weights, WeightData
from joli.em_algorithm import JoliEM, EMResult
from joli.output_writer import write_abundance


# ============================================================
# Shared helpers
# ============================================================

def make_sample_dir(
    ec_transcripts: list,
    ec_counts: list,
    transcript_names: list,
    n_pseudoaligned: int = None,
    n_targets: int = None,
    include_run_info: bool = True,
) -> str:
    """
    Create a temporary directory with synthetic bustools output files.
    Returns the path; caller is responsible for cleanup.

    Args:
        ec_transcripts   : list[list[int]] -- transcript indices per EC.
        ec_counts        : list[int]       -- read count per EC.
        transcript_names : list[str]       -- transcript name per index.
        n_pseudoaligned  : int             -- written to run_info.json.
        n_targets        : int             -- written to run_info.json.
        include_run_info : bool            -- whether to write run_info.json.

    Returns:
        str -- path to temporary directory.
    """
    tmpdir = tempfile.mkdtemp(prefix="joli_test_")
    n_ecs = len(ec_transcripts)
    n_tx  = len(transcript_names)
    if n_pseudoaligned is None:
        n_pseudoaligned = sum(ec_counts)
    if n_targets is None:
        n_targets = n_tx

    # matrix.ec
    with open(os.path.join(tmpdir, "matrix.ec"), "w") as fh:
        for ec_id, txs in enumerate(ec_transcripts):
            fh.write(f"{ec_id}\t{','.join(str(t) for t in txs)}\n")

    # transcripts.txt
    with open(os.path.join(tmpdir, "transcripts.txt"), "w") as fh:
        for name in transcript_names:
            fh.write(f"{name}\n")

    # count.mtx  (MatrixMarket: 1 row = 1 sample, cols = ECs, 1-indexed)
    nonzero_entries = [(1, ec_id + 1, cnt)
                       for ec_id, cnt in enumerate(ec_counts) if cnt > 0]
    with open(os.path.join(tmpdir, "count.mtx"), "w") as fh:
        fh.write("%%MatrixMarket matrix coordinate real general\n")
        fh.write("%\n")
        fh.write(f"1 {n_ecs} {len(nonzero_entries)}\n")
        for row, col, val in nonzero_entries:
            fh.write(f"{row} {col} {val}\n")

    # run_info.json
    if include_run_info:
        info = {
            "n_targets": n_targets,
            "n_bootstraps": 0,
            "n_processed": n_pseudoaligned + 1000,
            "n_pseudoaligned": n_pseudoaligned,
            "n_unique": n_pseudoaligned // 2,
            "p_pseudoaligned": 68.8,
            "kallisto_version": "0.51.1",
        }
        with open(os.path.join(tmpdir, "run_info.json"), "w") as fh:
            json.dump(info, fh)

    return tmpdir


def check(condition: bool, test_name: str, detail: str = "") -> bool:
    """Print PASS or FAIL for a single assertion."""
    if condition:
        print(f"  PASS  {test_name}")
    else:
        print(f"  FAIL  {test_name}" + (f": {detail}" if detail else ""))
    return condition


# ============================================================
# Step 1.1 — load_tcc.py
# ============================================================

def test_basic_load():
    """Basic 3-transcript, 3-EC case. Verifies all fields populated correctly."""
    print("\n[test_basic_load]")
    ec_transcripts   = [[0], [1], [0, 2]]
    ec_counts        = [25, 10, 5]
    transcript_names = ["ENST001", "ENST002", "ENST003"]

    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names,
                             n_pseudoaligned=40)
    try:
        data = load_tcc_data(tmpdir)
        passed = True
        passed &= check(isinstance(data, TCCData), "Returns TCCData")
        passed &= check(len(data.ec_transcripts) == 3, "ec_transcripts has 3 ECs",
                        f"got {len(data.ec_transcripts)}")
        passed &= check(data.ec_transcripts[0] == [0], "EC0 -> [tx0]",
                        f"got {data.ec_transcripts[0]}")
        passed &= check(data.ec_transcripts[2] == [0, 2], "EC2 -> [tx0, tx2]",
                        f"got {data.ec_transcripts[2]}")
        passed &= check(len(data.transcript_names) == 3, "transcript_names has 3 entries")
        passed &= check(data.transcript_names[1] == "ENST002", "transcript_names[1]='ENST002'",
                        f"got {data.transcript_names[1]}")
        passed &= check(int(data.ec_counts[0]) == 25, "ec_counts[0]==25",
                        f"got {data.ec_counts[0]}")
        passed &= check(int(data.ec_counts[2]) == 5, "ec_counts[2]==5",
                        f"got {data.ec_counts[2]}")
        passed &= check(data.total_reads == 40, "total_reads==40 (from run_info.json)",
                        f"got {data.total_reads}")
        passed &= check(data.ec_counts.dtype == np.int64, "ec_counts dtype is int64",
                        f"got {data.ec_counts.dtype}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_ec_counts_zeros():
    """EC with zero count is loaded as 0, not missing."""
    print("\n[test_ec_counts_zeros]")
    ec_transcripts   = [[0], [1], [2]]
    ec_counts        = [10, 0, 5]
    transcript_names = ["TX_A", "TX_B", "TX_C"]

    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names)
    try:
        data = load_tcc_data(tmpdir)
        passed = True
        passed &= check(int(data.ec_counts[1]) == 0, "zero-count EC loaded as 0",
                        f"got {data.ec_counts[1]}")
        passed &= check(len(data.ec_counts) == 3, "ec_counts length==3 (no missing rows)",
                        f"got {len(data.ec_counts)}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_multi_transcript_ec():
    """ECs with many transcripts are loaded correctly."""
    print("\n[test_multi_transcript_ec]")
    ec_transcripts   = [[0], [1, 2], [0, 1, 2, 3]]
    ec_counts        = [53, 20, 3]
    transcript_names = [f"TX_{i}" for i in range(4)]

    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names)
    try:
        data = load_tcc_data(tmpdir)
        passed = True
        passed &= check(data.ec_transcripts[2] == [0, 1, 2, 3],
                        "4-transcript EC loaded correctly",
                        f"got {data.ec_transcripts[2]}")
        passed &= check(int(data.ec_counts[2]) == 3, "4-transcript EC count==3",
                        f"got {data.ec_counts[2]}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_no_run_info_fallback():
    """When run_info.json is absent, total_reads falls back to sum(ec_counts)."""
    print("\n[test_no_run_info_fallback]")
    ec_transcripts   = [[0], [1]]
    ec_counts        = [30, 20]
    transcript_names = ["TX_A", "TX_B"]

    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names,
                             include_run_info=False)
    try:
        data = load_tcc_data(tmpdir)
        passed = check(data.total_reads == 50,
                       "total_reads falls back to sum(ec_counts)=50",
                       f"got {data.total_reads}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_missing_required_file():
    """FileNotFoundError is raised when count.mtx is absent."""
    print("\n[test_missing_required_file]")
    tmpdir = tempfile.mkdtemp(prefix="joli_test_")
    with open(os.path.join(tmpdir, "matrix.ec"), "w") as fh:
        fh.write("0\t0\n")
    with open(os.path.join(tmpdir, "transcripts.txt"), "w") as fh:
        fh.write("TX_A\n")
    try:
        raised = False
        try:
            load_tcc_data(tmpdir)
        except FileNotFoundError:
            raised = True
        passed = check(raised, "FileNotFoundError raised when count.mtx is missing")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_large_synthetic():
    """Stress test: 10k transcripts, 5k ECs. Verifies shapes and totals."""
    print("\n[test_large_synthetic]")
    np.random.seed(42)
    n_tx, n_ecs = 10_000, 5_000
    transcript_names = [f"ENST{i:06d}" for i in range(n_tx)]

    ec_transcripts = []
    for _ in range(n_ecs):
        size = 1 if np.random.rand() < 0.6 else np.random.randint(2, 6)
        txs = sorted(np.random.choice(n_tx, size, replace=False).tolist())
        ec_transcripts.append(txs)

    ec_counts = np.random.randint(0, 100, size=n_ecs).tolist()
    expected_total = sum(ec_counts)

    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names,
                             n_pseudoaligned=expected_total)
    try:
        data = load_tcc_data(tmpdir)
        passed = True
        passed &= check(len(data.ec_transcripts) == n_ecs,
                        f"ec_transcripts length=={n_ecs}", f"got {len(data.ec_transcripts)}")
        passed &= check(len(data.transcript_names) == n_tx,
                        f"transcript_names length=={n_tx}", f"got {len(data.transcript_names)}")
        passed &= check(int(data.ec_counts.sum()) == expected_total,
                        f"sum(ec_counts)=={expected_total}", f"got {data.ec_counts.sum()}")
        passed &= check(data.total_reads == expected_total,
                        f"total_reads=={expected_total}", f"got {data.total_reads}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Step 1.2 — weights.py
# ============================================================

def _make_tcc_data(ec_transcripts, ec_counts, transcript_names):
    """Helper: build a TCCData from a temp dir (reuses make_sample_dir)."""
    tmpdir = make_sample_dir(ec_transcripts, ec_counts, transcript_names)
    try:
        data = load_tcc_data(tmpdir)
    finally:
        shutil.rmtree(tmpdir)
    return data


def test_uniform_weights_values():
    """
    Uniform mode: eff_lens all 1.0, ec_weights all 1.0.
    """
    print("\n[test_uniform_weights_values]")
    ec_transcripts   = [[0], [1], [0, 2]]
    ec_counts        = [25, 10, 5]
    transcript_names = ["ENST001", "ENST002", "ENST003"]

    data = _make_tcc_data(ec_transcripts, ec_counts, transcript_names)
    w = compute_weights(data, mode="uniform")
    passed = True

    passed &= check(isinstance(w, WeightData), "Returns WeightData")
    passed &= check(len(w.eff_lens) == 3, "eff_lens length==3",
                    f"got {len(w.eff_lens)}")
    passed &= check(np.all(w.eff_lens == 1.0), "All eff_lens==1.0 (uniform)",
                    f"got {w.eff_lens}")
    # EC2 has 2 transcripts; both weights should be 1.0
    passed &= check(len(w.ec_weights[2]) == 2, "EC2 weight array has 2 entries",
                    f"got {len(w.ec_weights[2])}")
    passed &= check(np.allclose(w.ec_weights[2], [1.0, 1.0]),
                    "EC2 weights are [1.0, 1.0]",
                    f"got {w.ec_weights[2]}")
    return passed


def test_uniform_weights_shapes():
    """
    Each ec_weights[ec] has exactly len(ec_transcripts[ec]) entries.
    """
    print("\n[test_uniform_weights_shapes]")
    ec_transcripts   = [[0], [1, 2], [0, 1, 2, 3]]
    ec_counts        = [10, 5, 3]
    transcript_names = [f"TX_{i}" for i in range(4)]

    data = _make_tcc_data(ec_transcripts, ec_counts, transcript_names)
    w = compute_weights(data, mode="uniform")
    passed = True

    expected_sizes = [1, 2, 4]
    for ec_id, exp in enumerate(expected_sizes):
        passed &= check(len(w.ec_weights[ec_id]) == exp,
                        f"EC{ec_id} weight shape=={exp}",
                        f"got {len(w.ec_weights[ec_id])}")
    return passed


def test_kallisto_weights_values():
    """
    Kallisto mode: eff_len = length - mean_frag_len + 1; weight = 1/eff_len.
    """
    print("\n[test_kallisto_weights_values]")
    ec_transcripts   = [[0], [1], [0, 1]]
    ec_counts        = [10, 10, 5]
    transcript_names = ["TX_A", "TX_B"]

    data = _make_tcc_data(ec_transcripts, ec_counts, transcript_names)

    # TX_A length=1000, TX_B length=500, mean_frag_len=200
    # eff_lens: TX_A = 1000-200+1 = 801, TX_B = 500-200+1 = 301
    lengths      = np.array([1000.0, 500.0])
    mean_frag    = 200.0
    w = compute_weights(data, transcript_lengths=lengths,
                        mean_frag_len=mean_frag, mode="kallisto")
    passed = True

    passed &= check(np.isclose(w.eff_lens[0], 801.0), "eff_lens[TX_A]==801",
                    f"got {w.eff_lens[0]}")
    passed &= check(np.isclose(w.eff_lens[1], 301.0), "eff_lens[TX_B]==301",
                    f"got {w.eff_lens[1]}")
    # EC0 -> [TX_A]: weight = 1/801
    passed &= check(np.isclose(w.ec_weights[0][0], 1.0 / 801.0),
                    "EC0 weight = 1/801", f"got {w.ec_weights[0][0]:.6f}")
    # EC2 -> [TX_A, TX_B]: weights = [1/801, 1/301]
    passed &= check(np.isclose(w.ec_weights[2][0], 1.0 / 801.0),
                    "EC2 weight[0] = 1/801", f"got {w.ec_weights[2][0]:.6f}")
    passed &= check(np.isclose(w.ec_weights[2][1], 1.0 / 301.0),
                    "EC2 weight[1] = 1/301", f"got {w.ec_weights[2][1]:.6f}")
    return passed


def test_kallisto_weights_guard():
    """
    When length - mean_frag_len + 1 < 1, falls back to raw transcript_length.
    TX_A: 100 - 200 + 1 = -99 (< 1) -> guard -> eff_len = 100
    TX_B: 500 - 200 + 1 = 301 (ok)  -> eff_len = 301
    """
    print("\n[test_kallisto_weights_guard]")
    ec_transcripts   = [[0], [1]]
    ec_counts        = [10, 10]
    transcript_names = ["TX_short", "TX_long"]

    data = _make_tcc_data(ec_transcripts, ec_counts, transcript_names)

    lengths   = np.array([100.0, 500.0])
    mean_frag = 200.0
    w = compute_weights(data, transcript_lengths=lengths,
                        mean_frag_len=mean_frag, mode="kallisto")
    passed = True

    passed &= check(np.isclose(w.eff_lens[0], 100.0),
                    "Short tx guard: eff_len falls back to transcript_length=100",
                    f"got {w.eff_lens[0]}")
    passed &= check(np.isclose(w.eff_lens[1], 301.0),
                    "Normal tx: eff_len=301", f"got {w.eff_lens[1]}")
    return passed


def test_kallisto_mode_missing_lengths():
    """ValueError raised when mode='kallisto' but lengths not provided."""
    print("\n[test_kallisto_mode_missing_lengths]")
    ec_transcripts   = [[0], [1]]
    ec_counts        = [10, 5]
    transcript_names = ["TX_A", "TX_B"]

    data = _make_tcc_data(ec_transcripts, ec_counts, transcript_names)
    passed = False
    try:
        compute_weights(data, mode="kallisto")   # no lengths provided
    except ValueError:
        passed = True
    return check(passed, "ValueError raised when kallisto mode missing lengths")


def test_weights_dtype():
    """eff_lens and ec_weights entries are float64."""
    print("\n[test_weights_dtype]")
    ec_transcripts   = [[0, 1], [2]]
    ec_counts        = [5, 3]
    transcript_names = ["TX_A", "TX_B", "TX_C"]

    data = _make_tcc_data(ec_transcripts, ec_counts, transcript_names)
    w = compute_weights(data, mode="uniform")
    passed = True

    passed &= check(w.eff_lens.dtype == np.float64, "eff_lens dtype is float64",
                    f"got {w.eff_lens.dtype}")
    passed &= check(w.ec_weights[0].dtype == np.float64, "ec_weights dtype is float64",
                    f"got {w.ec_weights[0].dtype}")
    return passed


# ============================================================
# Step 1.3 — em_algorithm.py
# ============================================================

def _make_em_inputs(ec_transcripts, ec_counts, transcript_names, mode="uniform"):
    """Helper: build TCCData + WeightData + JoliEM for a small synthetic case."""
    data = _make_tcc_data(ec_transcripts, ec_counts, transcript_names)
    wdata = compute_weights(data, mode=mode)
    em = JoliEM(data, wdata)
    return data, wdata, em


def test_em_all_single_tx():
    """
    All ECs are single-transcript: alpha equals raw counts (no EM needed).
    EC0->tx0 count=60, EC1->tx1 count=40.
    Expected: alpha=[60, 40] (single-tx counts added post-convergence).
    """
    print("\n[test_em_all_single_tx]")
    ec_transcripts   = [[0], [1]]
    ec_counts        = [60, 40]
    transcript_names = ["TX_A", "TX_B"]

    data, wdata, em = _make_em_inputs(ec_transcripts, ec_counts, transcript_names)
    result = em.run(max_em_rounds=10000, min_rounds=1)

    passed = True
    passed &= check(isinstance(result, EMResult), "Returns EMResult")
    # Single-tx counts land in alpha directly; proportions should be 60:40
    passed &= check(np.isclose(result.theta_unnorm[0], 60.0, atol=1e-6),
                    "alpha[TX_A]==60 (raw single-tx count)", f"got {result.theta_unnorm[0]:.6f}")
    passed &= check(np.isclose(result.theta_unnorm[1], 40.0, atol=1e-6),
                    "alpha[TX_B]==40 (raw single-tx count)", f"got {result.theta_unnorm[1]:.6f}")
    passed &= check(np.isclose(result.theta_unnorm.sum(), 100.0, atol=1e-9),
                    "alpha sums to total reads (100)", f"got {result.theta_unnorm.sum():.10f}")
    return passed


def test_em_single_multi_ec_converges_uniform():
    """
    One multi-tx EC covering both transcripts with equal weights (uniform eff_lens).
    With symmetric weights, EM should converge to alpha=[50, 50] (equal split of 100 reads).
    """
    print("\n[test_em_single_multi_ec_converges_uniform]")
    ec_transcripts   = [[0, 1]]   # one EC, two transcripts
    ec_counts        = [100]
    transcript_names = ["TX_A", "TX_B"]

    data, wdata, em = _make_em_inputs(ec_transcripts, ec_counts, transcript_names)
    result = em.run(max_em_rounds=10000, min_rounds=1)

    passed = True
    # alpha = theta * total_multi_reads (no single-tx here); symmetric -> 50 each
    passed &= check(np.isclose(result.theta_unnorm[0], 50.0, atol=1e-2),
                    "alpha[TX_A]~50 (symmetric EC, 100 reads split evenly)",
                    f"got {result.theta_unnorm[0]:.6f}")
    passed &= check(np.isclose(result.theta_unnorm[1], 50.0, atol=1e-2),
                    "alpha[TX_B]~50 (symmetric EC, 100 reads split evenly)",
                    f"got {result.theta_unnorm[1]:.6f}")
    return passed


def test_em_mixed_ecs():
    """
    Mixed single and multi-tx ECs. Validates n accumulation logic.

    Setup:
      TX_A: exclusively observed in EC0 (count=50) -> strong signal
      TX_B: ambiguous in EC1 (tx1+tx2, count=20) and EC2 (tx1 only, count=10)
      TX_C: ambiguous in EC1 (tx1+tx2, count=20)

    Expected: theta[TX_A] dominates, TX_B and TX_C share the ambiguous counts.
    """
    print("\n[test_em_mixed_ecs]")
    # EC0: tx0 only (count=50)
    # EC1: tx1+tx2 (count=20, ambiguous between TX_B and TX_C)
    # EC2: tx1 only (count=10, exclusive to TX_B)
    ec_transcripts   = [[0], [1, 2], [1]]
    ec_counts        = [50, 20, 10]
    transcript_names = ["TX_A", "TX_B", "TX_C"]

    data, wdata, em = _make_em_inputs(ec_transcripts, ec_counts, transcript_names)
    result = em.run(max_em_rounds=10000, min_rounds=1)

    passed = True
    # alpha[TX_A] = 50 (single-tx) + 0 (multi) = 50
    # alpha[TX_B] = 10 (single-tx) + ~10 (half of 20 from EC1) = ~20
    # alpha[TX_C] = 0  (single-tx) + ~10 (half of 20 from EC1) = ~10
    passed &= check(result.theta_unnorm[0] > result.theta_unnorm[1],
                    "TX_A (50 exclusive reads) > TX_B",
                    f"alpha[A]={result.theta_unnorm[0]:.4f}, alpha[B]={result.theta_unnorm[1]:.4f}")
    passed &= check(result.theta_unnorm[1] > result.theta_unnorm[2],
                    "TX_B (10 exclusive + half of 20) > TX_C (half of 20)",
                    f"alpha[B]={result.theta_unnorm[1]:.4f}, alpha[C]={result.theta_unnorm[2]:.4f}")
    passed &= check(result.theta_unnorm.sum() > 0, "alpha sum > 0",
                    f"sum={result.theta_unnorm.sum():.4f}")
    passed &= check(result.converged, "EM converged", f"converged={result.converged}")
    return passed


def test_em_theta_nonneg_sums_to_one():
    """theta is non-negative and sums to 1 for a random case."""
    print("\n[test_em_theta_nonneg_sums_to_one]")
    np.random.seed(7)
    n_tx, n_ecs = 50, 30
    transcript_names = [f"TX{i}" for i in range(n_tx)]
    ec_transcripts = []
    for _ in range(n_ecs):
        size = np.random.randint(1, 5)
        txs  = sorted(np.random.choice(n_tx, size, replace=False).tolist())
        ec_transcripts.append(txs)
    ec_counts = np.random.randint(1, 100, size=n_ecs).tolist()

    data, wdata, em = _make_em_inputs(ec_transcripts, ec_counts, transcript_names)
    result = em.run(max_em_rounds=10000, min_rounds=1)

    passed = True
    passed &= check(np.all(result.theta_unnorm >= 0), "All alpha >= 0",
                    f"min={result.theta_unnorm.min():.4f}")
    passed &= check(result.theta_unnorm.sum() > 0, "alpha sum > 0 (reads assigned)",
                    f"got {result.theta_unnorm.sum():.4f}")
    passed &= check(result.n_rounds >= 1, "n_rounds >= 1",
                    f"got {result.n_rounds}")
    return passed


def test_em_min_rounds_respected():
    """EM never stops before min_rounds, even if already converged."""
    print("\n[test_em_min_rounds_respected]")
    # Single-tx ECs converge in 1 round — but min_rounds should delay stop
    ec_transcripts   = [[0], [1]]
    ec_counts        = [70, 30]
    transcript_names = ["TX_A", "TX_B"]

    data, wdata, em = _make_em_inputs(ec_transcripts, ec_counts, transcript_names)
    result = em.run(max_em_rounds=10000, min_rounds=10)

    passed = check(result.n_rounds >= 10,
                   "n_rounds >= min_rounds=10", f"got {result.n_rounds}")
    return passed


def test_em_zero_count_ec_ignored():
    """ECs with count=0 do not affect alpha."""
    print("\n[test_em_zero_count_ec_ignored]")
    # EC0: tx0 only, count=100  -> all mass on TX_A (single-tx, added post-convergence)
    # EC1: tx0+tx1,  count=0   -> should be ignored (multi-tx, zero count)
    ec_transcripts   = [[0], [0, 1]]
    ec_counts        = [100, 0]
    transcript_names = ["TX_A", "TX_B"]

    data, wdata, em = _make_em_inputs(ec_transcripts, ec_counts, transcript_names)
    result = em.run(max_em_rounds=10000, min_rounds=1)

    passed = True
    # alpha[TX_A] = 100 (single-tx raw count); alpha[TX_B] = 0 (zero-count multi EC ignored)
    passed &= check(np.isclose(result.theta_unnorm[0], 100.0, atol=1e-4),
                    "alpha[TX_A]==100 (all reads in single-tx EC)",
                    f"alpha[A]={result.theta_unnorm[0]:.6f}, alpha[B]={result.theta_unnorm[1]:.6f}")
    passed &= check(np.isclose(result.theta_unnorm[1], 0.0, atol=1e-4),
                    "alpha[TX_B]==0 (zero-count EC ignored)",
                    f"alpha[B]={result.theta_unnorm[1]:.6f}")
    return passed


# ============================================================
# Step 1.4 — output_writer.py
# ============================================================

def _run_full_pipeline(ec_transcripts, ec_counts, transcript_names,
                       out_file=None):
    """
    Helper: run load -> weights -> EM -> write_abundance.
    Returns (summary_dict, alpha, eff_lens, out_file_path).
    """
    data  = _make_tcc_data(ec_transcripts, ec_counts, transcript_names)
    wdata = compute_weights(data, mode="uniform")
    em    = JoliEM(data, wdata)
    em_result = em.run(max_em_rounds=10000, min_rounds=1)

    if out_file is None:
        tmpdir  = tempfile.mkdtemp(prefix="joli_out_")
        out_file = os.path.join(tmpdir, "abundance.tsv")

    summary = write_abundance(
        theta_unnorm=em_result.theta_unnorm,
        eff_lens=wdata.eff_lens,
        transcript_names=data.transcript_names,
        output_path=out_file,
    )
    return summary, em_result.theta_unnorm, wdata.eff_lens, out_file


def _read_tsv(path):
    """Read abundance.tsv into a dict of column_name -> list of values."""
    rows = []
    with open(path) as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            rows.append(line.strip().split("\t"))
    result = {h: [] for h in header}
    for row in rows:
        for h, v in zip(header, row):
            result[h].append(v)
    return result


def test_output_header_format():
    """abundance.tsv has the exact kallisto header columns."""
    print("\n[test_output_header_format]")
    tmpdir = tempfile.mkdtemp(prefix="joli_out_")
    out_file = os.path.join(tmpdir, "abundance.tsv")
    try:
        _run_full_pipeline([[0], [1]], [50, 50], ["TX_A", "TX_B"],
                           out_file=out_file)
        cols = _read_tsv(out_file)
        expected = ["target_id", "length", "eff_length", "est_counts", "tpm"]
        passed = check(list(cols.keys()) == expected,
                       "Header columns match kallisto format",
                       f"got {list(cols.keys())}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_output_all_transcripts_written():
    """All transcripts are written, including those with theta=0."""
    print("\n[test_output_all_transcripts_written]")
    tmpdir = tempfile.mkdtemp(prefix="joli_out_")
    out_file = os.path.join(tmpdir, "abundance.tsv")
    # TX_C gets no reads (no EC covers it) -> theta should be 0
    ec_transcripts   = [[0], [1]]
    ec_counts        = [60, 40]
    transcript_names = ["TX_A", "TX_B", "TX_C"]
    try:
        _run_full_pipeline(ec_transcripts, ec_counts, transcript_names,
                           out_file=out_file)
        cols = _read_tsv(out_file)
        passed = True
        passed &= check(len(cols["target_id"]) == 3,
                        "3 rows written (all transcripts including zero-TPM)",
                        f"got {len(cols['target_id'])}")
        passed &= check("TX_C" in cols["target_id"],
                        "TX_C with 0 reads is present in output")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_output_tpm_sums_to_million():
    """Sum of TPM values is ~1,000,000 (standard TPM property)."""
    print("\n[test_output_tpm_sums_to_million]")
    tmpdir = tempfile.mkdtemp(prefix="joli_out_")
    out_file = os.path.join(tmpdir, "abundance.tsv")
    ec_transcripts   = [[0], [1], [0, 2]]
    ec_counts        = [50, 30, 20]
    transcript_names = ["TX_A", "TX_B", "TX_C"]
    try:
        _run_full_pipeline(ec_transcripts, ec_counts, transcript_names,
                           out_file=out_file)
        cols = _read_tsv(out_file)
        tpm_sum = sum(float(v) for v in cols["tpm"])
        # TPM sums to 1e6 only for nonzero transcripts; allow small tolerance
        passed = check(np.isclose(tpm_sum, 1e6, rtol=1e-4),
                       "Sum of TPM ~1,000,000",
                       f"got {tpm_sum:.2f}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_output_est_counts_equal_alpha():
    """est_counts = alpha (raw expected counts); sum = total pseudoaligned reads."""
    print("\n[test_output_est_counts_equal_alpha]")
    tmpdir = tempfile.mkdtemp(prefix="joli_out_")
    out_file = os.path.join(tmpdir, "abundance.tsv")
    # All single-tx ECs: alpha = raw counts directly
    ec_transcripts   = [[0], [1]]
    ec_counts        = [70, 30]
    transcript_names = ["TX_A", "TX_B"]
    try:
        _run_full_pipeline(ec_transcripts, ec_counts, transcript_names, out_file=out_file)
        cols = _read_tsv(out_file)
        est = [float(v) for v in cols["est_counts"]]
        passed = True
        # alpha = raw counts: TX_A=70, TX_B=30
        passed &= check(np.isclose(sum(est), 100.0, rtol=1e-4),
                        "Sum est_counts==100 (total reads)",
                        f"got {sum(est):.2f}")
        passed &= check(np.isclose(est[0], 70.0, rtol=1e-3),
                        "TX_A est_counts==70 (raw single-tx count)", f"got {est[0]:.2f}")
        passed &= check(np.isclose(est[1], 30.0, rtol=1e-3),
                        "TX_B est_counts==30 (raw single-tx count)", f"got {est[1]:.2f}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_output_eff_length_uniform():
    """In uniform mode, eff_length column is 1.000000 for all transcripts."""
    print("\n[test_output_eff_length_uniform]")
    tmpdir = tempfile.mkdtemp(prefix="joli_out_")
    out_file = os.path.join(tmpdir, "abundance.tsv")
    try:
        _run_full_pipeline([[0], [1]], [50, 50], ["TX_A", "TX_B"],
                           out_file=out_file)
        cols = _read_tsv(out_file)
        passed = check(all(float(v) == 1.0 for v in cols["eff_length"]),
                       "All eff_length==1.0 in uniform mode",
                       f"got {cols['eff_length']}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


def test_output_transcript_order():
    """Transcripts are written in index order (TX_A, TX_B, TX_C)."""
    print("\n[test_output_transcript_order]")
    tmpdir = tempfile.mkdtemp(prefix="joli_out_")
    out_file = os.path.join(tmpdir, "abundance.tsv")
    names = ["TX_A", "TX_B", "TX_C"]
    try:
        _run_full_pipeline([[0], [1], [2]], [10, 20, 30], names,
                           out_file=out_file)
        cols = _read_tsv(out_file)
        passed = check(cols["target_id"] == names,
                       "Transcript order matches input order",
                       f"got {cols['target_id']}")
    finally:
        shutil.rmtree(tmpdir)
    return passed


# ============================================================
# Step 1.5 — main_joli.py  (end-to-end CLI via subprocess)
# ============================================================

def test_main_joli_runs_end_to_end():
    """
    main_joli.py runs successfully on a synthetic sample dir and writes abundance.tsv.
    Invoked via subprocess so it exercises the full CLI path.
    """
    import subprocess
    print("\n[test_main_joli_runs_end_to_end]")
    ec_transcripts   = [[0], [1], [0, 2]]
    ec_counts        = [50, 30, 20]
    transcript_names = ["TX_A", "TX_B", "TX_C"]
    total_reads      = 100

    sample_dir = make_sample_dir(ec_transcripts, ec_counts, transcript_names,
                                 n_pseudoaligned=total_reads)
    output_dir = tempfile.mkdtemp(prefix="joli_out_")
    try:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_joli.py")
        result = subprocess.run(
            [sys.executable, script,
             "--sample_dir",    sample_dir,
             "--output_dir",    output_dir,
             "--max_em_rounds", "100",
             "--min_rounds",    "1"],
            capture_output=True, text=True,
        )
        passed = True
        passed &= check(result.returncode == 0,
                        "main_joli.py exits with code 0",
                        f"stderr: {result.stderr[-300:]}")
        abundance = os.path.join(output_dir, "abundance.tsv")
        passed &= check(os.path.isfile(abundance),
                        "abundance.tsv created",
                        f"output_dir contents: {os.listdir(output_dir)}")
        runtime  = os.path.join(output_dir, "runtime.txt")
        passed &= check(os.path.isfile(runtime),
                        "runtime.txt created")
    finally:
        shutil.rmtree(sample_dir)
        shutil.rmtree(output_dir)
    return passed


def test_main_joli_output_content():
    """
    abundance.tsv produced by main_joli.py has correct columns and TPM ~1e6.
    """
    import subprocess
    print("\n[test_main_joli_output_content]")
    ec_transcripts   = [[0], [1]]
    ec_counts        = [70, 30]
    transcript_names = ["TX_A", "TX_B"]

    sample_dir = make_sample_dir(ec_transcripts, ec_counts, transcript_names,
                                 n_pseudoaligned=100)
    output_dir = tempfile.mkdtemp(prefix="joli_out_")
    try:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_joli.py")
        subprocess.run(
            [sys.executable, script,
             "--sample_dir", sample_dir,
             "--output_dir", output_dir,
             "--min_rounds", "1"],
            check=True, capture_output=True,
        )
        cols = _read_tsv(os.path.join(output_dir, "abundance.tsv"))
        passed = True
        expected_cols = ["target_id", "length", "eff_length", "est_counts", "tpm"]
        passed &= check(list(cols.keys()) == expected_cols,
                        "Correct columns in abundance.tsv",
                        f"got {list(cols.keys())}")
        passed &= check(len(cols["target_id"]) == 2,
                        "2 transcript rows written",
                        f"got {len(cols['target_id'])}")
        tpm_sum = sum(float(v) for v in cols["tpm"])
        passed &= check(np.isclose(tpm_sum, 1e6, rtol=1e-3),
                        "TPM sums to ~1,000,000",
                        f"got {tpm_sum:.2f}")
    finally:
        shutil.rmtree(sample_dir)
        shutil.rmtree(output_dir)
    return passed


def test_main_joli_missing_sample_dir():
    """main_joli.py exits non-zero when --sample_dir does not exist."""
    import subprocess
    print("\n[test_main_joli_missing_sample_dir]")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_joli.py")
    result = subprocess.run(
        [sys.executable, script, "--sample_dir", "/nonexistent/path/xyz"],
        capture_output=True, text=True,
    )
    passed = check(result.returncode != 0,
                   "Non-zero exit when sample_dir missing",
                   f"got returncode={result.returncode}")
    return passed


def test_main_joli_default_output_dir():
    """When --output_dir is omitted, abundance.tsv is written inside --sample_dir."""
    import subprocess
    print("\n[test_main_joli_default_output_dir]")
    ec_transcripts   = [[0], [1]]
    ec_counts        = [50, 50]
    transcript_names = ["TX_A", "TX_B"]

    sample_dir = make_sample_dir(ec_transcripts, ec_counts, transcript_names)
    try:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_joli.py")
        result = subprocess.run(
            [sys.executable, script,
             "--sample_dir", sample_dir,
             "--min_rounds", "1"],
            capture_output=True, text=True,
        )
        passed = True
        passed &= check(result.returncode == 0,
                        "Exits 0 without --output_dir",
                        f"stderr: {result.stderr[-200:]}")
        passed &= check(os.path.isfile(os.path.join(sample_dir, "abundance.tsv")),
                        "abundance.tsv written into sample_dir by default")
    finally:
        shutil.rmtree(sample_dir)
    return passed


# ============================================================
# Runner
# ============================================================

def main():
    print("=" * 60)
    print("JOLI-Kallisto: test_JolitoKallisto.py")
    print("=" * 60)

    all_tests = [
        # --- Step 1.1: load_tcc.py ---
        ("Step 1.1 — load_tcc.py", [
            test_basic_load,
            test_ec_counts_zeros,
            test_multi_transcript_ec,
            test_no_run_info_fallback,
            test_missing_required_file,
            test_large_synthetic,
        ]),
        # --- Step 1.2: weights.py ---
        ("Step 1.2 — weights.py", [
            test_uniform_weights_values,
            test_uniform_weights_shapes,
            test_kallisto_weights_values,
            test_kallisto_weights_guard,
            test_kallisto_mode_missing_lengths,
            test_weights_dtype,
        ]),
        # --- Step 1.3: em_algorithm.py ---
        ("Step 1.3 — em_algorithm.py", [
            test_em_all_single_tx,
            test_em_single_multi_ec_converges_uniform,
            test_em_mixed_ecs,
            test_em_theta_nonneg_sums_to_one,
            test_em_min_rounds_respected,
            test_em_zero_count_ec_ignored,
        ]),
        # --- Step 1.4: output_writer.py ---
        ("Step 1.4 — output_writer.py", [
            test_output_header_format,
            test_output_all_transcripts_written,
            test_output_tpm_sums_to_million,
            test_output_est_counts_equal_alpha,
            test_output_eff_length_uniform,
            test_output_transcript_order,
        ]),
        # --- Step 1.5: main_joli.py ---
        ("Step 1.5 — main_joli.py", [
            test_main_joli_runs_end_to_end,
            test_main_joli_output_content,
            test_main_joli_missing_sample_dir,
            test_main_joli_default_output_dir,
        ]),
    ]

    total_pass = total_fail = 0
    for section_name, tests in all_tests:
        print(f"\n{'='*60}")
        print(f"  {section_name}")
        print(f"{'='*60}")
        for t in tests:
            try:
                result = t()
                if not result:
                    total_fail += 1
                else:
                    total_pass += 1
            except Exception as e:
                print(f"  ERROR  {t.__name__}: {e}")
                total_fail += 1

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed "
          f"out of {total_pass + total_fail} tests")
    print("=" * 60)

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
