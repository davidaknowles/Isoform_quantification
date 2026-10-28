"""
plot_metric_curves.py
=====================
Plot GT metric training curves (Spearman, Pearson, MAE, RMSE) across all
training iterations for JK multi-sample (MAP EM) and JK single-sample (plain EM).

One figure per sample (sim1, sim2).
Each figure: 5-row GridSpec layout.
  Rows 0–3 : 4 GT metrics × 3 universe columns (Spearman / Pearson / MAE / RMSE)
  Row 4    : mean |Δθ| convergence curve spanning all 3 columns
  Columns  : All transcripts | Active (GT ∪ nonzero pred) | GT non-zero only
  Lines    : JK MS (solid blue) | JK Single (dashed red)
  x-axis   : iteration / round number
  y-axis   : metric value (row 4: log scale)

Snapshot formats
----------------
JK MS  ({JK_MS_EXP}/snapshots.pkl):
    {
      "sample_names": list[str],
      "snapshots": [{"round": int, "alpha": ndarray(T,), "thetas": [ndarray(T,), ...]}, ...]
    }

JK single  ({JK_SS_EXP}/{sample}/theta_snapshots.pkl):
    {
      "transcript_names": list[str],
      "snapshots": [(round_num: int, theta: ndarray(T,)), ...]
    }

Inputs:
    - snapshots.pkl       from JK_MS_EXP
    - theta_snapshots.pkl from JK_SS_EXP/{sample}/
    - abundance.tsv       from each exp/{sample}/ (for eff_length)
    - PB_sample{1,2}_gt.tsv from GT_BASE

Outputs (Rule 3 — 1 file per sample → figures/ folder):
    {JK_MS_EXP}/figures/metric_curves_sim1_{timestamp}.png
    {JK_MS_EXP}/figures/metric_curves_sim2_{timestamp}.png

Run:
    cd /gpfs/commons/home/atalukder/RNA_Splicing/code/JOLI_Kallisto
    conda activate Joli_kallisto
    python analysis/plot_metric_curves.py
"""

from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load core/ modules (no __init__.py needed)
from joli.load_tcc import load_tcc_data  # noqa: E402

# ============================================================
# CONFIG — edit here; do not edit below
# ============================================================

RESULTS_BASE = "/gpfs/commons/home/atalukder/RNA_Splicing/files/results"

# JK_MS_EXP  = "exprmnt_2026_03_30__22_37_55"   # JK multi-sample MAP EM
# JK_SS_EXP  = "exprmnt_2026_03_30__22_41_19"   # JK single-sample plain EM

JK_MS_EXP  = "exprmnt_2026_05_21__11_28_30"   # JK multi-sample MAP EM
JK_SS_EXP  = "exprmnt_2026_05_18__12_46_25"   # JK single-sample plain EM

GT_BASE = "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/sim_real_data/ground_truths"

SAMPLE_GT_MAP = {
    "sim1": "PB_sample1_gt.tsv",
    "sim2": "PB_sample2_gt.tsv",
}

# Bustools output directories for each sample (used to load TCC data so that
# single-tx raw counts can be added to snapshot theta before TPM conversion).
# These are the --sample_dirs passed to main_multisample_joli.py at run time.
# Both JK MS and JK SS were run on the same reads, so one set of dirs covers both.
SAMPLE_DIRS = {
    "sim1": "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/sim_real_data/kallisto_output/ds_100_num1_aln_01_long",
    "sim2": "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/sim_real_data/kallisto_output/ds_100_num1_aln_21_long",
}

# ============================================================
# END CONFIG
# ============================================================

TIMESTAMP = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

METRICS = ["spearman", "pearson", "mae", "rmse"]
METRIC_LABELS = {
    "spearman": "Spearman Correlation",
    "pearson":  "Pearson Correlation",
    "mae":      "MAE",
    "rmse":     "RMSE",
}
UNIVERSES = ["all", "active", "gt"]
UNIVERSE_LABELS = {
    "all":    "All transcripts",
    "active": "Active (GT ∪ nonzero pred)",
    "gt":     "GT non-zero only",
}

# When True: reconstruct full alpha = theta * total_multi_reads + single_tx_raw_counts
# before TPM conversion, matching output_writer.py exactly.
# When False: use normalized theta directly (faster; omits single-tx contribution).
INCLUDE_SINGLE_TX_CORRECTION = True # ********FALSE option DOES NOT WORK


# ============================================================
# Metric helpers (mirror of run_snapshot_gt_comparison.py)
# ============================================================

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation between two arrays.

    Args:
        x : np.ndarray -- first array.
        y : np.ndarray -- second array.

    Returns:
        float -- Pearson r, or NaN if degenerate.
    """
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Spearman correlation between two arrays.

    Args:
        x : np.ndarray -- first array.
        y : np.ndarray -- second array.

    Returns:
        float -- Spearman r, or NaN if degenerate.
    """
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(xr, yr)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mean absolute error.

    Args:
        x : np.ndarray -- predicted.
        y : np.ndarray -- ground truth.

    Returns:
        float -- MAE.
    """
    return float(np.mean(np.abs(x - y)))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute root mean squared error.

    Args:
        x : np.ndarray -- predicted.
        y : np.ndarray -- ground truth.

    Returns:
        float -- RMSE.
    """
    return float(np.sqrt(np.mean((x - y) ** 2)))


def compute_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute Spearman, Pearson, MAE, RMSE between two TPM arrays.

    Args:
        x : np.ndarray -- predicted TPM values.
        y : np.ndarray -- ground truth TPM values.

    Returns:
        dict with keys: spearman, pearson, mae, rmse.
    """
    return {
        "spearman": spearman_corr(x, y),
        "pearson":  pearson_corr(x, y),
        "mae":      mae(x, y),
        "rmse":     rmse(x, y),
    }


# ============================================================
# I/O helpers (mirror of run_snapshot_gt_comparison.py)
# ============================================================

def read_ground_truth(path: Path) -> pd.DataFrame:
    """
    Read a ground truth CSV/TSV file.

    Handles both comma-separated (PB_sample*_gt.tsv) and tab-separated variants.
    Sniffs delimiter from first line.

    Args:
        path : Path -- path to ground truth file.

    Returns:
        pd.DataFrame with columns [transcript_id, tpm_gt].
    """
    with open(path) as fh:
        first_line = fh.readline()
    sep = "," if first_line.count(",") > first_line.count("\t") else "\t"
    df = pd.read_csv(path, sep=sep, index_col=0)

    id_candidates  = ["transcript_name", "transcript_id", "target_id", "Name"]
    val_candidates = ["tpm", "TPM", "est_counts"]
    id_col  = next((c for c in id_candidates  if c in df.columns), df.columns[0])
    val_col = next((c for c in val_candidates if c in df.columns), df.columns[1])

    out = df[[id_col, val_col]].copy()
    out.columns = ["transcript_id", "tpm_gt"]
    out["transcript_id"] = out["transcript_id"].astype(str)
    out["tpm_gt"] = pd.to_numeric(out["tpm_gt"], errors="coerce").fillna(0.0)
    return out


def read_eff_lens(abundance_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Read transcript names and effective lengths from an existing abundance.tsv.

    Args:
        abundance_path : Path -- path to {exp}/{sample}/abundance.tsv.

    Returns:
        tuple(list[str], np.ndarray): transcript_names, eff_lens (float64, T).
    """
    df = pd.read_csv(abundance_path, sep="\t", comment="#")
    id_col  = "target_id"  if "target_id"  in df.columns else df.columns[0]
    len_col = "eff_length" if "eff_length" in df.columns else df.columns[2]
    transcript_names = df[id_col].astype(str).tolist()
    eff_lens         = pd.to_numeric(df[len_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    return transcript_names, eff_lens


def load_single_tx_data(sample_dir: str, n_transcripts: int) -> tuple[float, np.ndarray]:
    """
    Load TCC data and return single-tx raw counts per transcript.

    This is the missing piece that em_step omits but em.run() adds before writing
    abundance.tsv (see em_algorithm.py lines 497-501):
        alpha = theta * total_multi_reads
        alpha[single_tx_ids] += single_ec_counts

    Args:
        sample_dir   : str -- bustools output directory for this sample.
        n_transcripts: int -- total number of transcripts (T).

    Returns:
        tuple:
          total_multi_reads   (float)          -- total reads from multi-tx ECs.
          single_tx_raw_counts(np.ndarray, T)  -- per-transcript single-tx raw counts.
    """
    print(f"  Loading TCC data from: {sample_dir}")
    tcc = load_tcc_data(sample_dir)

    single_tx_raw_counts = np.zeros(n_transcripts, dtype=np.float64)
    total_single = 0.0

    for ec_id, tx_list in enumerate(tcc.ec_transcripts):
        if len(tx_list) == 1:
            t_idx = tx_list[0]
            if t_idx < n_transcripts:
                single_tx_raw_counts[t_idx] += tcc.ec_counts[ec_id]
            total_single += tcc.ec_counts[ec_id]

    total_multi_reads = float(tcc.total_reads - total_single)
    print(f"    total_reads={tcc.total_reads}, "
          f"total_multi={total_multi_reads:.0f}, "
          f"total_single={total_single:.0f}, "
          f"single_tx_transcripts={(single_tx_raw_counts > 0).sum()}")
    return total_multi_reads, single_tx_raw_counts


def theta_to_tpm(
    theta:                np.ndarray,
    eff_lens:             np.ndarray,
    total_multi_reads:    float        = None,
    single_tx_raw_counts: np.ndarray  = None,
) -> np.ndarray:
    """
    Convert snapshot theta to TPM, optionally matching output_writer.py exactly.

    Snapshot theta (from em_step) covers only multi-tx EC assignments.
    When correction is enabled (total_multi_reads + single_tx_raw_counts provided),
    reconstructs full alpha and computes TPM as output_writer.py does:
        alpha = theta * total_multi_reads + single_tx_raw_counts
        tpm   = (alpha / eff_len) / sum(alpha / eff_len) * 1e6

    When correction is disabled (both None), uses normalized theta directly:
        tpm = (theta / eff_len) / sum(theta / eff_len) * 1e6

    Args:
        theta                : np.ndarray (T,) -- normalized snapshot theta.
        eff_lens             : np.ndarray (T,) -- effective transcript lengths.
        total_multi_reads    : float | None    -- sum of ec_counts for multi-tx ECs.
                                                 None = skip correction.
        single_tx_raw_counts : np.ndarray | None -- per-transcript single-tx raw counts.
                                                 None = skip correction.

    Returns:
        np.ndarray (T,) -- TPM values (sums to ~1e6).
    """
    if total_multi_reads is not None and single_tx_raw_counts is not None:
        # Corrected: reconstruct full alpha (matches output_writer.py)
        scale = theta * total_multi_reads + single_tx_raw_counts
    else:
        # Uncorrected: use normalized theta directly
        scale = theta

    safe_eff = np.where(eff_lens > 0, eff_lens, 1.0)
    rho      = scale / safe_eff
    rho_sum  = rho.sum()
    return (rho / rho_sum * 1e6) if rho_sum > 0 else np.zeros_like(rho)


def compare_to_gt(
    tpm_pred:         np.ndarray,
    transcript_names: list[str],
    gt_df:            pd.DataFrame,
) -> dict[str, dict]:
    """
    Compare a TPM prediction array against ground truth across 3 universes.

    Universes:
      1. all    — outer join (includes TN pairs)
      2. active — GT ∪ nonzero pred (excludes TN pairs)
      3. gt     — GT non-zero only

    Args:
        tpm_pred         : np.ndarray   -- predicted TPM (T,).
        transcript_names : list[str]    -- transcript name at each index.
        gt_df            : pd.DataFrame -- columns [transcript_id, tpm_gt].

    Returns:
        dict[universe_key -> dict of metric_name -> float].
    """
    pred_df = pd.DataFrame({
        "transcript_id": transcript_names,
        "tpm_pred":      tpm_pred,
    })

    # Universe 1: all (outer join)
    merged_all = pred_df.merge(gt_df, on="transcript_id", how="outer")
    merged_all["tpm_pred"] = merged_all["tpm_pred"].fillna(0.0)
    merged_all["tpm_gt"]   = merged_all["tpm_gt"].fillna(0.0)
    x_all = merged_all["tpm_pred"].to_numpy()
    y_all = merged_all["tpm_gt"].to_numpy()

    # Universe 2: active (GT ∪ nonzero pred), excludes TN
    pred_nz_df    = pred_df[pred_df["tpm_pred"] > 0]
    merged_active = pred_nz_df.merge(gt_df, on="transcript_id", how="outer")
    merged_active["tpm_pred"] = merged_active["tpm_pred"].fillna(0.0)
    merged_active["tpm_gt"]   = merged_active["tpm_gt"].fillna(0.0)
    x_active = merged_active["tpm_pred"].to_numpy()
    y_active = merged_active["tpm_gt"].to_numpy()

    # Universe 3: GT non-zero only
    gt_nz_mask = merged_all["tpm_gt"] > 0
    x_gt = merged_all.loc[gt_nz_mask, "tpm_pred"].to_numpy()
    y_gt = merged_all.loc[gt_nz_mask, "tpm_gt"].to_numpy()

    return {
        "all":    compute_metrics(x_all,    y_all),
        "active": compute_metrics(x_active, y_active),
        "gt":     compute_metrics(x_gt,     y_gt),
    }


# ============================================================
# Snapshot loaders
# ============================================================

def load_ms_snapshots(
    ms_exp_dir: Path,
    sample: str,
) -> tuple[list[int], list[np.ndarray]]:
    """
    Load JK multi-sample snapshots for one sample.

    Note: transcript names are NOT returned — use abundance.tsv names (same
    ordering, correct GT-matching format), mirroring run_snapshot_gt_comparison.py.

    Args:
        ms_exp_dir : Path -- top-level JK MS experiment directory.
        sample     : str  -- sample name (e.g. "sim1").

    Returns:
        tuple:
          rounds (list[int])       -- round number per snapshot
          thetas (list[ndarray T]) -- theta at each round for this sample
    """
    pkl_path = ms_exp_dir / "snapshots.pkl"
    print(f"  Loading JK MS snapshots from {pkl_path}")
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    sample_names = data["sample_names"]
    if sample not in sample_names:
        raise ValueError(f"Sample '{sample}' not found in snapshots. Available: {sample_names}")
    s_idx = sample_names.index(sample)

    rounds = []
    thetas = []
    for snap in data["snapshots"]:
        rounds.append(snap["round"])
        thetas.append(snap["thetas"][s_idx])

    print(f"    Sample '{sample}': {len(rounds)} snapshots, T={len(thetas[0]) if thetas else 0}")
    return rounds, thetas


def load_ss_snapshots(
    ss_exp_dir: Path,
    sample: str,
) -> tuple[list[int], list[np.ndarray]]:
    """
    Load JK single-sample theta snapshots for one sample.

    Note: transcript names are intentionally NOT returned from the pkl —
    always use names from abundance.tsv (same ordering, correct GT-matching
    format), mirroring run_snapshot_gt_comparison.py line 745.

    Args:
        ss_exp_dir : Path -- top-level JK SS experiment directory.
        sample     : str  -- sample name (e.g. "sim1").

    Returns:
        tuple:
          rounds (list[int])       -- round number per snapshot
          thetas (list[ndarray T]) -- theta at each round
    """
    pkl_path = ss_exp_dir / sample / "theta_snapshots.pkl"
    print(f"  Loading JK single snapshots from {pkl_path}")
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    rounds = [snap[0] for snap in data["snapshots"]]
    thetas = [snap[1] for snap in data["snapshots"]]

    print(f"    Sample '{sample}': {len(rounds)} snapshots, T={len(thetas[0]) if thetas else 0}")
    return rounds, thetas


# ============================================================
# Core: compute metric curves for one sample + one method
# ============================================================

def compute_metric_curves(
    rounds:                list[int],
    thetas:                list[np.ndarray],
    transcript_names:      list[str],
    eff_lens:              np.ndarray,
    total_multi_reads:     float,
    single_tx_raw_counts:  np.ndarray,
    gt_df:                 pd.DataFrame,
    label:                 str,
) -> tuple[dict[str, dict[str, list]], list[float]]:
    """
    For every snapshot, compute GT metrics and mean absolute delta-theta.

    Delta-theta: mean(|theta_current - theta_previous|) between consecutive snapshots.
    This mirrors the EM stopping criterion (both SS and MS use relative theta change;
    the absolute version plotted here shows the same convergence trend).
    First snapshot has delta=NaN (no previous state).

    Args:
        rounds               : list[int]       -- round indices.
        thetas               : list[ndarray]   -- theta at each round.
        transcript_names     : list[str]       -- transcript name order.
        eff_lens             : np.ndarray (T,) -- effective lengths.
        total_multi_reads    : float | None    -- total reads from multi-tx ECs.
        single_tx_raw_counts : np.ndarray | None -- per-transcript single-tx raw counts.
        gt_df                : pd.DataFrame    -- [transcript_id, tpm_gt].
        label                : str             -- method label for printing.

    Returns:
        tuple:
          curves (dict[universe -> dict[metric -> list]])  -- GT metrics per round.
          deltas (list[float])                             -- mean |Δθ| per round.
    """
    curves: dict[str, dict[str, list]] = {
        univ: {m: [] for m in METRICS} for univ in UNIVERSES
    }
    deltas: list[float] = []
    prev_theta: np.ndarray | None = None

    for i, (rnd, theta) in enumerate(zip(rounds, thetas)):
        theta_f = theta.astype(np.float64)

        # Delta theta: mean absolute change from previous snapshot
        if prev_theta is not None:
            delta = float(np.mean(np.abs(theta_f - prev_theta)))
        else:
            delta = float("nan")
        deltas.append(delta)
        prev_theta = theta_f

        tpm = theta_to_tpm(theta_f, eff_lens, total_multi_reads, single_tx_raw_counts)
        result = compare_to_gt(tpm, transcript_names, gt_df)
        for univ in UNIVERSES:
            for m in METRICS:
                curves[univ][m].append(result[univ][m])

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{label}] round {rnd}: "
                  f"spearman(all)={result['all']['spearman']:.4f}, "
                  f"pearson(all)={result['all']['pearson']:.4f}, "
                  f"mean|Δθ|={delta:.2e}")

    return curves, deltas


# ============================================================
# Plotting
# ============================================================

def plot_sample(
    sample:     str,
    ms_rounds:  list[int],
    ms_curves:  dict,
    ms_deltas:  list[float],
    ss_rounds:  list[int],
    ss_curves:  dict,
    ss_deltas:  list[float],
    out_path:   Path,
) -> None:
    """
    Create a 5-row figure for one sample and save to out_path.

    Layout (GridSpec):
      Row 0-3 : 4 GT metrics × 3 universe columns (Spearman / Pearson / MAE / RMSE)
      Row 4   : mean |Δθ| convergence curve spanning all 3 columns

    Columns (rows 0-3): All transcripts | Active | GT non-zero
    Lines: JK MS (solid blue) | JK Single (dashed red)

    Args:
        sample    : str        -- sample name (for title).
        ms_rounds : list[int]  -- JK MS round numbers.
        ms_curves : dict       -- JK MS GT metric curves.
        ms_deltas : list[float]-- JK MS mean |Δθ| per round.
        ss_rounds : list[int]  -- JK Single round numbers.
        ss_curves : dict       -- JK Single GT metric curves.
        ss_deltas : list[float]-- JK Single mean |Δθ| per round.
        out_path  : Path       -- where to save the figure.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 14))
    gs  = GridSpec(5, 3, figure=fig, hspace=0.5, wspace=0.35)
    fig.suptitle(f"GT Metric Curves — {sample}", fontsize=13)

    # --- Rows 0–3: GT metrics × universes ---
    for row_idx, metric in enumerate(METRICS):
        for col_idx, univ in enumerate(UNIVERSES):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            ax.plot(ms_rounds, ms_curves[univ][metric],
                    color="#1565C0", linewidth=1.5, linestyle="-", label="JK MS")
            ax.plot(ss_rounds, ss_curves[univ][metric],
                    color="#C62828", linewidth=1.5, linestyle="--", label="JK Single")

            ax.set_xlabel("Iteration", fontsize=8)
            ax.tick_params(labelsize=7)

            if row_idx == 0:
                ax.set_title(UNIVERSE_LABELS[univ], fontsize=9, pad=4)
            if col_idx == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="best")

    # --- Row 4: delta-theta (spans all 3 columns) ---
    ax_delta = fig.add_subplot(gs[4, :])

    # Plot from round index 1 onward (first round has NaN delta)
    ms_rounds_valid = [r for r, d in zip(ms_rounds, ms_deltas) if not np.isnan(d)]
    ms_deltas_valid = [d for d in ms_deltas if not np.isnan(d)]
    ss_rounds_valid = [r for r, d in zip(ss_rounds, ss_deltas) if not np.isnan(d)]
    ss_deltas_valid = [d for d in ss_deltas if not np.isnan(d)]

    ax_delta.plot(ms_rounds_valid, ms_deltas_valid,
                  color="#1565C0", linewidth=1.5, linestyle="-", label="JK MS")
    ax_delta.plot(ss_rounds_valid, ss_deltas_valid,
                  color="#C62828", linewidth=1.5, linestyle="--", label="JK Single")

    ax_delta.set_xlabel("Iteration", fontsize=9)
    ax_delta.set_ylabel("Mean |Δθ|\n(consecutive snapshots)", fontsize=9)
    ax_delta.set_title(
        "Convergence: mean absolute θ change between snapshots\n"
        "(EM stops when per-transcript relative |Δθ|/θ < 1% for all θ > 0.01)",
        fontsize=8,
    )
    ax_delta.set_yscale("log")   # log scale shows convergence clearly
    ax_delta.legend(fontsize=8, loc="upper right")
    ax_delta.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ============================================================
# Main
# ============================================================

def run():
    """
    Load snapshots for each sample, compute GT metric curves, save figures.

    Run:
        cd /gpfs/commons/home/atalukder/RNA_Splicing/code/JOLI_Kallisto
        conda activate Joli_kallisto
        python analysis/plot_metric_curves.py
    """
    ms_base = Path(RESULTS_BASE) / JK_MS_EXP
    ss_base = Path(RESULTS_BASE) / JK_SS_EXP
    fig_dir = ms_base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for sample, gt_file in SAMPLE_GT_MAP.items():
        print(f"\n{'='*60}")
        print(f"Sample: {sample}")
        print(f"{'='*60}")

        # Load ground truth
        gt_path = Path(GT_BASE) / gt_file
        print(f"  Loading GT from {gt_path}")
        gt_df = read_ground_truth(gt_path)
        print(f"  GT nonzero entries: {(gt_df['tpm_gt'] > 0).sum()}")

        # Load transcript names and effective lengths from abundance.tsv for BOTH
        # methods — same approach as run_snapshot_gt_comparison.py (lines 668, 745).
        # The transcript names in abundance.tsv match the GT file IDs; pkl
        # transcript_names may differ in format and must NOT be used for GT merging.
        abundance_ms = ms_base / sample / "abundance.tsv"
        ms_tx_names, ms_eff_lens = read_eff_lens(abundance_ms)
        print(f"  JK MS eff_lens loaded: T={len(ms_tx_names)}")

        abundance_ss = ss_base / sample / "abundance.tsv"
        ss_tx_names, ss_eff_lens = read_eff_lens(abundance_ss)
        print(f"  JK SS eff_lens loaded: T={len(ss_tx_names)}")

        # Load snapshots (transcript names come from abundance.tsv, not pkl)
        ms_rounds, ms_thetas = load_ms_snapshots(ms_base, sample)
        ss_rounds, ss_thetas    = load_ss_snapshots(ss_base, sample)

        # Sanity check: theta shape must match eff_lens length
        if ms_thetas and len(ms_thetas[0]) != len(ms_tx_names):
            raise ValueError(
                f"JK MS snapshot T={len(ms_thetas[0])} != abundance T={len(ms_tx_names)} "
                f"for {sample}. Cannot proceed."
            )
        if ss_thetas and len(ss_thetas[0]) != len(ss_tx_names):
            raise ValueError(
                f"JK SS snapshot T={len(ss_thetas[0])} != abundance T={len(ss_tx_names)} "
                f"for {sample}. Cannot proceed."
            )

        # Optionally load TCC data for single-tx correction (controlled by CONFIG switch).
        # em_step omits single-tx reads; output_writer.py adds them back.
        # Without correction: TPM is computed from normalized theta only (fast, biased).
        # With correction   : TPM matches abundance.tsv exactly (recommended).
        if INCLUDE_SINGLE_TX_CORRECTION:
            sample_dir = SAMPLE_DIRS[sample]
            total_multi_reads, single_tx_raw_counts = load_single_tx_data(
                sample_dir, n_transcripts=len(ms_tx_names)
            )
            assert len(single_tx_raw_counts) == len(ss_tx_names), (
                f"single_tx_raw_counts T={len(single_tx_raw_counts)} "
                f"!= JK SS T={len(ss_tx_names)}"
            )
        else:
            print(f"  [single-tx correction OFF] using normalized theta for TPM")
            total_multi_reads    = None
            single_tx_raw_counts = None

        # Compute metric curves
        print(f"\n  Computing JK MS metric curves ({len(ms_rounds)} snapshots)...")
        ms_curves, ms_deltas = compute_metric_curves(
            ms_rounds, ms_thetas, ms_tx_names, ms_eff_lens,
            total_multi_reads, single_tx_raw_counts, gt_df, label=f"JK MS {sample}"
        )

        print(f"\n  Computing JK Single metric curves ({len(ss_rounds)} snapshots)...")
        ss_curves, ss_deltas = compute_metric_curves(
            ss_rounds, ss_thetas, ss_tx_names, ss_eff_lens,
            total_multi_reads, single_tx_raw_counts, gt_df, label=f"JK SS {sample}"
        )

        # Plot and save
        out_path = fig_dir / f"metric_curves_{sample}_{TIMESTAMP}.png"
        print(f"\n  Plotting {sample}...")
        plot_sample(
            sample,
            ms_rounds, ms_curves, ms_deltas,
            ss_rounds, ss_curves, ss_deltas,
            out_path,
        )

    print(f"\nDone. Figures saved to: {fig_dir}")


if __name__ == "__main__":
    run()
