"""
plot_convergence_animation.py
=============================
Visualise isoform quantification convergence across training rounds for three
methods, comparing against ground truth (GT):

  - JK     : JOLI single-sample plain EM (from theta_snapshots.pkl)
  - JK MS  : JOLI multi-sample MAP EM  (from snapshots.pkl)

One animated GIF is produced per sample (sim1, sim2).  Each GIF has four panels:

  ┌──────────────────────┬──────────────────────┐
  │  Panel 1: GT (static)│  Panel 2: JK theta   │
  ├──────────────────────┼──────────────────────┤
  │  Panel 3: JK MS alpha│  Panel 4: JK MS theta│
  └──────────────────────┴──────────────────────┘

Transcript universe per sample:
  - TP : GT TPM > 0  AND  predicted non-zero (either method)  →  top portion
  - FP : GT TPM = 0  AND  predicted non-zero (either method)  →  bottom (below separator)
  - FN (GT>0, pred=0) are excluded.

Each panel is a scatter plot:
  x-axis = transcript rank (fixed, sorted by GT TPM desc for TP; pred TPM desc for FP)
  y-axis = TPM / theta / alpha value (log-scale for GT panel; linear for predictions)
  color  = value (coolwarm; normalized per panel across all frames for stable comparison)

Also produces:
  - jk_inter_sample_spearman.png    : Spearman(theta_sim1, theta_sim2) vs round for JK
  - jkms_inter_sample_spearman.png  : Spearman(theta_sim1, theta_sim2) vs round for JK MS

All outputs are saved inside JK_MS_DIR.

Inputs:
  JK_SINGLE_SIM1_DIR : experiment folder from main_joli.py for sim1 (with theta_snapshots.pkl)
  JK_SINGLE_SIM2_DIR : experiment folder from main_joli.py for sim2 (with theta_snapshots.pkl)
  JK_MS_DIR          : experiment folder from main_multisample_joli.py (with snapshots.pkl)
  GT_SIM1_PATH       : ground truth TSV for sim1
  GT_SIM2_PATH       : ground truth TSV for sim2

Run:
    cd /gpfs/commons/home/atalukder/RNA_Splicing/code/JOLI_Kallisto
    conda activate NanoCount_5
    python analysis/plot_convergence_animation.py
"""

import os
import pickle
import sys
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIG — edit before running; do not edit below
# ============================================================

RESULTS_BASE = "/gpfs/commons/home/atalukder/RNA_Splicing/files/results"

# Experiment folders (names only; combined with RESULTS_BASE)
JK_SINGLE_SIM1_DIR = "exprmnt_2026_03_30__11_14_19"   # e.g. "exprmnt_2026_03_30__10_00_00"  (JK single, sim1)
JK_SINGLE_SIM2_DIR = "exprmnt_2026_03_30__11_14_19"   # e.g. "exprmnt_2026_03_30__10_05_00"  (JK single, sim2)
JK_MS_DIR          = "exprmnt_2026_03_30__11_39_16"   # e.g. "exprmnt_2026_03_30__10_10_00"  (JK MS)

# Sample names — must match the subfolder names inside JK_MS_DIR
SAMPLE_NAMES = ["sim1", "sim2"]

# Ground truth paths
GT_PATHS = {
    "sim1": "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/sim_real_data/ground_truths/PB_sample1_gt.tsv",
    "sim2": "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/sim_real_data/ground_truths/PB_sample2_gt.tsv",
}

# Animation settings
FPS               = 2      # frames per second (2 = 0.5 s/frame)
COLORMAP          = "coolwarm"
LOG_SCALE_GT      = True   # use log y-axis for GT panel (recommended for TPM)

# ============================================================
# END CONFIG
# ============================================================


# ============================================================
# I/O helpers
# ============================================================

def load_gt(path: str) -> pd.DataFrame:
    """
    Load a ground truth TSV/CSV file.

    Args:
        path : str -- path to GT file (CSV with unnamed index, transcript_name, tpm).

    Returns:
        pd.DataFrame with columns [transcript_id, tpm_gt].
    """
    df = pd.read_csv(path, index_col=0)
    id_col  = next((c for c in ["transcript_name", "transcript_id", "target_id"]
                    if c in df.columns), df.columns[0])
    val_col = next((c for c in ["tpm", "TPM"] if c in df.columns), df.columns[1])
    out = df[[id_col, val_col]].copy()
    out.columns = ["transcript_id", "tpm_gt"]
    out["transcript_id"] = out["transcript_id"].astype(str)
    out["tpm_gt"]        = pd.to_numeric(out["tpm_gt"], errors="coerce").fillna(0.0)
    return out


def load_jk_single_snapshots(exp_dir: str, sample_name: str) -> dict:
    """
    Load theta snapshots from a JK single-sample experiment folder.

    Looks for <exp_dir>/<sample_name>/theta_snapshots.pkl.

    Args:
        exp_dir     : str -- absolute path to the experiment folder.
        sample_name : str -- sample subfolder name (e.g. "sim1").

    Returns:
        dict with keys:
          "transcript_names" : list[str]
          "snapshots"        : list of (round_num, theta_array)
    """
    path = os.path.join(exp_dir, sample_name, "theta_snapshots.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"JK single snapshot not found: {path}\n"
            "Re-run main_joli.py with SAVE_SNAPSHOTS=True."
        )
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    print(f"  Loaded JK single snapshots ({len(data['snapshots'])} frames): {path}")
    return data


def load_jk_ms_snapshots(exp_dir: str) -> dict:
    """
    Load alpha + theta snapshots from a JK MS experiment folder.

    Looks for <exp_dir>/snapshots.pkl.

    Args:
        exp_dir : str -- absolute path to the experiment folder.

    Returns:
        dict with keys:
          "sample_names" : list[str]
          "snapshots"    : list of {"round", "alpha", "thetas": [theta_s0, theta_s1, ...]}
    """
    path = os.path.join(exp_dir, "snapshots.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"JK MS snapshot not found: {path}\n"
            "Re-run run_multisample_joli.sh with SAVE_SNAPSHOTS=true."
        )
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    print(f"  Loaded JK MS snapshots ({len(data['snapshots'])} frames): {path}")
    return data


def load_transcript_names(exp_dir: str, sample_name: str) -> list:
    """
    Load the ordered transcript name list from a JK MS experiment's transcripts.txt.

    Falls back to looking inside the sample subfolder's bustools output.

    Args:
        exp_dir     : str -- absolute path to the experiment folder.
        sample_name : str -- sample subfolder name.

    Returns:
        list[str] -- ordered transcript names matching the theta/alpha arrays.
    """
    # The experiment folder contains a code_snapshot; transcripts.txt is in the
    # bustools cache dir which is stored in experiment_description.log.
    # Simpler: load from the JK MS sample's abundance.tsv which has transcript IDs.
    abund_path = os.path.join(exp_dir, sample_name, "abundance.tsv")
    if os.path.exists(abund_path):
        df = pd.read_csv(abund_path, sep="\t")
        id_col = next((c for c in ["transcript_id", "target_id"] if c in df.columns),
                      df.columns[0])
        return list(df[id_col].astype(str))
    raise FileNotFoundError(
        f"Cannot find transcript names: abundance.tsv not found at {abund_path}"
    )


# ============================================================
# Universe builder
# ============================================================

def build_universe(
    gt_df:       pd.DataFrame,
    tx_names:    list,
    jk_theta_final:    np.ndarray,
    jkms_theta_final:  np.ndarray,
) -> pd.DataFrame:
    """
    Build the transcript universe for one sample: TP + FP, sorted for plotting.

    TP = GT TPM > 0 AND predicted non-zero in at least one method.
    FP = GT TPM = 0 AND predicted non-zero in at least one method.
    FN (GT>0, pred=0 in all methods) are excluded.

    Args:
        gt_df            : pd.DataFrame -- [transcript_id, tpm_gt].
        tx_names         : list[str]    -- ordered transcript names (index = array position).
        jk_theta_final   : np.ndarray   -- final JK single theta (T,).
        jkms_theta_final : np.ndarray   -- final JK MS theta for this sample (T,).

    Returns:
        pd.DataFrame with columns:
          transcript_id, tpm_gt, jk_theta, jkms_theta,
          is_tp (bool), is_fp (bool), rank (int, 0-indexed),
          separator_after (bool -- True for last TP row, marks the TP/FP boundary)
        Sorted: TP first (by tpm_gt desc), FP second (by max predicted desc).
    """
    n = len(tx_names)
    df = pd.DataFrame({
        "transcript_id": tx_names,
        "jk_theta":      jk_theta_final[:n],
        "jkms_theta":    jkms_theta_final[:n],
    })

    # Merge GT values
    df = df.merge(gt_df, on="transcript_id", how="left")
    df["tpm_gt"] = df["tpm_gt"].fillna(0.0)

    # GT-only universe: keep only transcripts where GT TPM > 0, sorted by GT TPM desc.
    df = df[df["tpm_gt"] > 0].copy()
    fp_df = fp_df.sort_values("_max_pred", ascending=False).drop(columns="_max_pred")

    universe = df.sort_values("tpm_gt", ascending=False).reset_index(drop=True)
    universe["rank"] = np.arange(len(universe))
    n_tp = len(universe)
    print(f"  Universe: {n_tp} GT transcripts (FP excluded)")
    return universe, n_tp


# ============================================================
# Frame rendering
# ============================================================

def _get_values_at_round(
    universe:    pd.DataFrame,
    tx_index:    dict,
    snapshots:   list,
    frame_idx:   int,
    col_key:     str,
) -> np.ndarray:
    """
    Extract predicted values from a snapshot at a given frame index,
    aligned to the universe transcript order.

    Args:
        universe  : pd.DataFrame -- universe with [transcript_id, rank].
        tx_index  : dict         -- {transcript_id: array_index}.
        snapshots : list         -- list of (round_num, theta) for JK single,
                                    or list of dicts for JK MS.
        frame_idx : int          -- index into snapshots list.
        col_key   : str          -- "jk_theta", "jkms_alpha", or "jkms_theta_s{N}".

    Returns:
        np.ndarray -- values aligned to universe.rank order.
    """
    snap = snapshots[frame_idx]

    if col_key == "jk_theta":
        # JK single: snap = (round_num, theta_array)
        _, theta = snap
        vals = np.array([theta[tx_index.get(tid, 0)] for tid in universe["transcript_id"]])

    elif col_key == "jkms_alpha":
        # JK MS: snap = {"round", "alpha", "thetas"}
        alpha = snap["alpha"]
        vals  = np.array([alpha[tx_index.get(tid, 0)] for tid in universe["transcript_id"]])
        # Normalize alpha to sum to 1 for comparable scale
        s = vals.sum()
        vals = vals / s if s > 0 else vals

    else:
        # jkms_theta_sN: e.g. "jkms_theta_s0"
        s_idx = int(col_key.split("_s")[-1])
        theta = snap["thetas"][s_idx]
        vals  = np.array([theta[tx_index.get(tid, 0)] for tid in universe["transcript_id"]])

    return vals


def render_frame(
    fig_size:       tuple,
    sample_name:    str,
    round_num:      int,
    universe:       pd.DataFrame,
    n_tp:           int,
    gt_vals:        np.ndarray,
    jk_vals:        np.ndarray,
    jkms_alpha_vals: np.ndarray,
    jkms_theta_vals: np.ndarray,
    gt_vmin:        float,
    gt_vmax:        float,
    jk_vmin:        float,
    jk_vmax:        float,
    alpha_vmin:     float,
    alpha_vmax:     float,
    jkms_vmin:      float,
    jkms_vmax:      float,
    log_scale_gt:   bool = True,
    cmap:           str  = "coolwarm",
) -> np.ndarray:
    """
    Render one GIF frame as a numpy RGB image array.

    Produces a 2×2 subplot figure with:
      Panel 1 (top-left)  : GT values (static reference)
      Panel 2 (top-right) : JK single-sample theta
      Panel 3 (bottom-left): JK MS alpha (normalized)
      Panel 4 (bottom-right): JK MS theta for this sample

    Args:
        fig_size         : tuple(float, float)  -- figure size in inches.
        sample_name      : str                  -- e.g. "sim1".
        round_num        : int                  -- current training round (for title).
        universe         : pd.DataFrame         -- sorted universe with ranks.
        n_tp             : int                  -- number of TP transcripts.
        gt_vals          : np.ndarray           -- GT TPM per universe position.
        jk_vals          : np.ndarray           -- JK theta per universe position.
        jkms_alpha_vals  : np.ndarray           -- normalized JK MS alpha.
        jkms_theta_vals  : np.ndarray           -- JK MS theta for this sample.
        {X}_vmin/vmax    : float                -- colormap bounds per panel.
        log_scale_gt     : bool                 -- log y-axis for GT panel.
        cmap             : str                  -- matplotlib colormap name.

    Returns:
        np.ndarray -- RGB image, shape (H, W, 3), dtype uint8.
    """
    ranks = universe["rank"].values
    sep   = n_tp - 0.5   # vertical separator between TP and FP regions

    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle(f"{sample_name}  —  Round {round_num}", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "GT (reference)",      gt_vals,         gt_vmin,     gt_vmax,     True),
        (axes[0, 1], "JK single-sample θ",  jk_vals,         jk_vmin,     jk_vmax,     False),
        (axes[1, 0], "JK MS  α (raw)",       jkms_alpha_vals, alpha_vmin,  alpha_vmax,  False),
        (axes[1, 1], "JK MS  θ",            jkms_theta_vals, jkms_vmin,   jkms_vmax,   False),
    ]

    norm_obj_cache = {}
    for ax, title, vals, vmin, vmax, is_gt in panels:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        norm_obj_cache[title] = norm
        sc = ax.scatter(ranks, vals, c=vals, cmap=cmap, norm=norm,
                        s=0.5, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)

        # Separator line between TP and FP regions
        if n_tp < len(ranks):
            ax.axvline(x=sep, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.text(sep + 1, vmax * 0.95 if not is_gt else vmax,
                    "FP →", fontsize=6, color="black", va="top")

        if is_gt and log_scale_gt:
            ax.set_yscale("log")

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Transcript rank (GT order)", fontsize=7)
        ax.set_ylabel("Value", fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )[..., :3]   # drop alpha channel → RGB
    plt.close(fig)
    return img


# ============================================================
# Per-panel vmin/vmax across all frames
# ============================================================

def compute_panel_bounds(vals_per_frame: list) -> tuple:
    """
    Compute stable vmin/vmax for a panel across all frames.

    Uses the 1st and 99th percentile of non-zero values across all frames
    to avoid a single extreme frame from dominating the colormap.

    Args:
        vals_per_frame : list[np.ndarray] -- one array per frame.

    Returns:
        tuple(float, float) -- (vmin, vmax).
    """
    all_vals = np.concatenate([v[v > 0] for v in vals_per_frame if (v > 0).any()])
    if len(all_vals) == 0:
        return 0.0, 1.0
    return float(np.percentile(all_vals, 1)), float(all_vals.max())


# ============================================================
# Spearman curve plots
# ============================================================

def plot_spearman_curve(
    rounds:     list,
    spearmans:  list,
    title:      str,
    xlabel:     str,
    output_path: str,
) -> None:
    """
    Plot inter-sample Spearman correlation vs training round and save as PNG.

    Args:
        rounds      : list[int]   -- round numbers (x-axis).
        spearmans   : list[float] -- Spearman values (y-axis).
        title       : str         -- plot title.
        xlabel      : str         -- x-axis label.
        output_path : str         -- path to save PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, spearmans, "b-o", markersize=3, linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Spearman (sim1 ↔ sim2)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved Spearman curve: {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Generate convergence animation GIFs and Spearman correlation curves.

    Steps:
      1. Print GT vs GT correlation (sanity check).
      2. Load JK single-sample snapshots for both samples.
      3. Load JK MS snapshots.
      4. For each sample: build universe, render frames, save GIF.
      5. Plot JK inter-sample Spearman vs round.
      6. Plot JK MS inter-sample Spearman vs round (from training_stats.pkl).

    Run:
        cd /gpfs/commons/home/atalukder/RNA_Splicing/code/JOLI_Kallisto
        conda activate NanoCount_5
        python analysis/plot_convergence_animation.py
    """
    # Validate CONFIG
    for name, val in [("JK_SINGLE_SIM1_DIR", JK_SINGLE_SIM1_DIR),
                      ("JK_SINGLE_SIM2_DIR", JK_SINGLE_SIM2_DIR),
                      ("JK_MS_DIR",          JK_MS_DIR)]:
        if not val:
            print(f"ERROR: {name} is empty. Set it in the CONFIG section.")
            sys.exit(1)

    # Resolve absolute paths
    jk_sim1_dir = os.path.join(RESULTS_BASE, JK_SINGLE_SIM1_DIR)
    jk_sim2_dir = os.path.join(RESULTS_BASE, JK_SINGLE_SIM2_DIR)
    jk_ms_dir   = os.path.join(RESULTS_BASE, JK_MS_DIR)

    for d in [jk_sim1_dir, jk_sim2_dir, jk_ms_dir]:
        if not os.path.isdir(d):
            print(f"ERROR: directory not found: {d}")
            sys.exit(1)

    # Create figures/ subfolder inside JK MS experiment dir
    figures_dir = os.path.join(jk_ms_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("plot_convergence_animation.py")
    print("=" * 70)
    print(f"  Output figures dir: {figures_dir}")

    # ----------------------------------------------------------
    # 1. GT vs GT correlation
    # ----------------------------------------------------------
    print("\n[1] GT vs GT correlation")
    gt_dfs = {s: load_gt(GT_PATHS[s]) for s in SAMPLE_NAMES}

    merged_gt = gt_dfs[SAMPLE_NAMES[0]].merge(
        gt_dfs[SAMPLE_NAMES[1]], on="transcript_id", how="inner",
        suffixes=("_s1", "_s2")
    )
    gt_sp = float(spearmanr(merged_gt["tpm_gt_s1"], merged_gt["tpm_gt_s2"]).statistic
                  if hasattr(spearmanr(merged_gt["tpm_gt_s1"], merged_gt["tpm_gt_s2"]), "statistic")
                  else spearmanr(merged_gt["tpm_gt_s1"], merged_gt["tpm_gt_s2"])[0])
    print(f"  GT sim1 vs GT sim2 Spearman : {gt_sp:.6f}")

    # ----------------------------------------------------------
    # 2. Load all snapshots
    # ----------------------------------------------------------
    print("\n[2] Loading snapshots")
    jk_snaps = {
        SAMPLE_NAMES[0]: load_jk_single_snapshots(jk_sim1_dir, SAMPLE_NAMES[0]),
        SAMPLE_NAMES[1]: load_jk_single_snapshots(jk_sim2_dir, SAMPLE_NAMES[1]),
    }
    jk_ms_snaps = load_jk_ms_snapshots(jk_ms_dir)

    # Synchronise frames: only rounds present in ALL sources
    jk_rounds_s1   = [r for r, _ in jk_snaps[SAMPLE_NAMES[0]]["snapshots"]]
    jk_rounds_s2   = [r for r, _ in jk_snaps[SAMPLE_NAMES[1]]["snapshots"]]
    jkms_rounds    = [s["round"] for s in jk_ms_snaps["snapshots"]]

    # Build common frame list: use JK MS rounds as reference
    # (JK single may have more rounds since it runs independently)
    common_rounds = sorted(set(jkms_rounds))
    print(f"  JK single sim1 frames : {len(jk_rounds_s1)}")
    print(f"  JK single sim2 frames : {len(jk_rounds_s2)}")
    print(f"  JK MS frames          : {len(jkms_rounds)}")
    print(f"  Common frames used    : {len(common_rounds)}")

    # Index lookups: round_num → snapshot index
    def make_round_index(snaps_list, key="round"):
        if isinstance(snaps_list[0], tuple):
            return {r: i for i, (r, _) in enumerate(snaps_list)}
        return {s[key]: i for i, s in enumerate(snaps_list)}

    jk_ridx = {
        SAMPLE_NAMES[0]: make_round_index(jk_snaps[SAMPLE_NAMES[0]]["snapshots"]),
        SAMPLE_NAMES[1]: make_round_index(jk_snaps[SAMPLE_NAMES[1]]["snapshots"]),
    }
    jkms_ridx = make_round_index(jk_ms_snaps["snapshots"])

    # Shared transcript name → array index mapping (from JK MS)
    jkms_sample_names = jk_ms_snaps["sample_names"]
    tx_names_ms = load_transcript_names(jk_ms_dir, jkms_sample_names[0])
    tx_index_ms = {tid: i for i, tid in enumerate(tx_names_ms)}

    # JK single transcript name → index (may differ per sample)
    tx_index_jk = {
        SAMPLE_NAMES[0]: {tid: i for i, tid in
                          enumerate(jk_snaps[SAMPLE_NAMES[0]]["transcript_names"])},
        SAMPLE_NAMES[1]: {tid: i for i, tid in
                          enumerate(jk_snaps[SAMPLE_NAMES[1]]["transcript_names"])},
    }

    # ----------------------------------------------------------
    # 3. Build GIF for each sample
    # ----------------------------------------------------------
    print("\n[3] Generating GIFs")

    # JK MS sample index for each sample name
    jkms_sidx = {s: jkms_sample_names.index(s) for s in SAMPLE_NAMES
                 if s in jkms_sample_names}

    for sample in SAMPLE_NAMES:
        print(f"\n  --- {sample} ---")
        gt_df   = gt_dfs[sample]
        s_idx   = jkms_sidx.get(sample, 0)

        # Final predicted thetas (last frame) for universe construction
        jk_final_snap  = jk_snaps[sample]["snapshots"][-1]
        jkms_final_snap = jk_ms_snaps["snapshots"][-1]
        jk_final_theta  = jk_final_snap[1]
        jkms_final_theta = jkms_final_snap["thetas"][s_idx]

        # Build universe using JK MS transcript names (shared index)
        universe, n_tp = build_universe(
            gt_df           = gt_df,
            tx_names        = tx_names_ms,
            jk_theta_final  = np.array([
                jk_final_theta[tx_index_jk[sample].get(tid, 0)]
                for tid in tx_names_ms
            ]),
            jkms_theta_final = jkms_final_theta,
        )

        gt_vals = np.array([
            gt_df.set_index("transcript_id")["tpm_gt"].get(tid, 0.0)
            for tid in universe["transcript_id"]
        ])

        # Pre-collect all frame values to compute stable vmin/vmax
        jk_all, alpha_all, jkms_all = [], [], []

        # Filter common_rounds to those available in both JK single and JK MS for this sample
        available_rounds = [r for r in common_rounds
                            if r in jk_ridx[sample] and r in jkms_ridx]
        print(f"  Frames to render: {len(available_rounds)}")

        for rnd in available_rounds:
            jk_fi   = jk_ridx[sample][rnd]
            jkms_fi = jkms_ridx[rnd]

            # JK theta aligned to universe
            _, jk_theta = jk_snaps[sample]["snapshots"][jk_fi]
            jk_vals_r = np.array([
                jk_theta[tx_index_jk[sample].get(tid, 0)]
                for tid in universe["transcript_id"]
            ])

            # JK MS alpha — raw Dirichlet concentration values (no normalization).
            # Normalizing by sum(alpha) is misleading because thousands of near-zero
            # transcripts inflate the denominator, giving tiny values (~5-8 TPM).
            alpha_r = jk_ms_snaps["snapshots"][jkms_fi]["alpha"]
            alpha_vals_r = np.array([alpha_r[tx_index_ms.get(tid, 0)]
                                     for tid in universe["transcript_id"]])

            # JK MS theta
            jkms_theta_r = jk_ms_snaps["snapshots"][jkms_fi]["thetas"][s_idx]
            jkms_vals_r  = np.array([jkms_theta_r[tx_index_ms.get(tid, 0)]
                                     for tid in universe["transcript_id"]])

            jk_all.append(jk_vals_r)
            alpha_all.append(alpha_vals_r)
            jkms_all.append(jkms_vals_r)

        # Compute panel-wide vmin/vmax.
        # Theta panels share GT bounds so all three are directly comparable.
        # Alpha uses its own bounds (raw Dirichlet concentration, independent scale).
        gt_vmin,    gt_vmax    = compute_panel_bounds([gt_vals])
        jk_vmin,    jk_vmax    = gt_vmin, gt_vmax
        jkms_vmin,  jkms_vmax  = gt_vmin, gt_vmax
        alpha_vmin, alpha_vmax = compute_panel_bounds(alpha_all)

        # Render frames
        frames = []
        for i, rnd in enumerate(available_rounds):
            img = render_frame(
                fig_size        = (14, 9),
                sample_name     = sample,
                round_num       = rnd,
                universe        = universe,
                n_tp            = n_tp,
                gt_vals         = gt_vals,
                jk_vals         = jk_all[i],
                jkms_alpha_vals = alpha_all[i],
                jkms_theta_vals = jkms_all[i],
                gt_vmin         = gt_vmin,
                gt_vmax         = gt_vmax,
                jk_vmin         = jk_vmin,
                jk_vmax         = jk_vmax,
                alpha_vmin      = alpha_vmin,
                alpha_vmax      = alpha_vmax,
                jkms_vmin       = jkms_vmin,
                jkms_vmax       = jkms_vmax,
                log_scale_gt    = LOG_SCALE_GT,
                cmap            = COLORMAP,
            )
            frames.append(img)
            if (i + 1) % 10 == 0:
                print(f"    Rendered {i + 1} / {len(available_rounds)} frames")

        # Save GIF
        gif_path = os.path.join(figures_dir, f"convergence_{sample}.gif")
        duration = 1.0 / FPS          # seconds per frame
        imageio.mimsave(gif_path, frames, duration=duration, loop=0)
        print(f"  Saved GIF ({len(frames)} frames, {FPS} fps): {gif_path}")

    # ----------------------------------------------------------
    # 4. JK single inter-sample Spearman curve
    # ----------------------------------------------------------
    print("\n[4] JK single inter-sample Spearman curve")
    jk_spearman_rounds, jk_spearmans = [], []

    s0, s1 = SAMPLE_NAMES[0], SAMPLE_NAMES[1]
    shared_jk_rounds = sorted(set(jk_ridx[s0].keys()) & set(jk_ridx[s1].keys()))

    for rnd in shared_jk_rounds:
        _, theta0 = jk_snaps[s0]["snapshots"][jk_ridx[s0][rnd]]
        _, theta1 = jk_snaps[s1]["snapshots"][jk_ridx[s1][rnd]]
        # Align to shared transcripts via JK MS index
        v0 = np.array([theta0[tx_index_jk[s0].get(tid, 0)] for tid in tx_names_ms])
        v1 = np.array([theta1[tx_index_jk[s1].get(tid, 0)] for tid in tx_names_ms])
        sp = float(spearmanr(v0, v1).statistic
                   if hasattr(spearmanr(v0, v1), "statistic")
                   else spearmanr(v0, v1)[0])
        jk_spearman_rounds.append(rnd)
        jk_spearmans.append(sp)

    plot_spearman_curve(
        rounds      = jk_spearman_rounds,
        spearmans   = jk_spearmans,
        title       = "JK single-sample: inter-sample Spearman vs EM round",
        xlabel      = "EM round",
        output_path = os.path.join(figures_dir, "jk_inter_sample_spearman.png"),
    )

    # ----------------------------------------------------------
    # 5. JK MS inter-sample Spearman curve (from training_stats.pkl)
    # ----------------------------------------------------------
    print("\n[5] JK MS inter-sample Spearman curve")
    stats_path = os.path.join(jk_ms_dir, "training_stats.pkl")
    if os.path.exists(stats_path):
        from joli.training_tracker import TrainingTracker
        tracker = TrainingTracker.load(stats_path)

        pair_key = (SAMPLE_NAMES[0], SAMPLE_NAMES[1])
        ms_rounds, ms_spearmans = [], []
        for rec in tracker.history:
            corr = rec["inter_sample_corr"].get(
                pair_key,
                rec["inter_sample_corr"].get((SAMPLE_NAMES[1], SAMPLE_NAMES[0]))
            )
            if corr is not None:
                ms_rounds.append(rec["gd_round"] + 1)
                ms_spearmans.append(corr["spearman"])

        plot_spearman_curve(
            rounds      = ms_rounds,
            spearmans   = ms_spearmans,
            title       = "JK MS: inter-sample Spearman vs training round",
            xlabel      = "Training round",
            output_path = os.path.join(figures_dir, "jkms_inter_sample_spearman.png"),
        )
    else:
        print(f"  training_stats.pkl not found at {stats_path}, skipping JK MS curve.")

    print("\n[Done] All outputs saved to:", jk_ms_dir)


if __name__ == "__main__":
    main()
