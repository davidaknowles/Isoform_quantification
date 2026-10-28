"""
plot_training.py
================
Generates training diagnostic figures from a TrainingTracker saved during a
multi-sample MAP EM run.

Figures produced (saved to figures/ inside the experiment folder):
  fig_gd_convergence.png      GD loss + alpha max change per round
  fig_em_convergence.png      EM rounds per sample + EM convergence rate per round
  fig_inter_sample_corr.png   Spearman + Pearson between all theta pairs
  fig_theta_vs_alpha_corr.png Spearman + Pearson each theta vs normalized alpha
  fig_alpha_stats.png         Alpha sum + alpha entropy per round
  fig_nonzero_transcripts.png Non-zero transcripts per sample per round

Can be called:
  (a) from main_multisample_joli.py after training — figures saved automatically.
  (b) standalone from command line to re-plot from a saved training_stats.pkl:

      conda activate NanoCount_5
      cd JOLI_Kallisto
      python analysis/plot_training.py \\
          --stats_path /path/to/exprmnt_.../training_stats.pkl \\
          --figures_dir /path/to/exprmnt_.../figures/

Inputs:
  TrainingTracker object (loaded from training_stats.pkl)

Outputs:
  .png figures in the specified figures directory
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for cluster/SLURM
import matplotlib.pyplot as plt
import numpy as np

# Allow standalone execution: add core/ to path for TrainingTracker import
from joli.training_tracker import TrainingTracker


# ============================================================
# Plotting helpers
# ============================================================

def _rounds(history: list) -> list:
    """Return list of 1-indexed GD round numbers."""
    return [rec["gd_round"] + 1 for rec in history]


def _save(fig, path: str) -> None:
    """Save figure and close it."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot_training] Saved: {path}")


# ============================================================
# Individual figure functions
# ============================================================

def plot_gd_convergence(history: list, figures_dir: str) -> None:
    """
    Figure 1: GD loss and alpha max change per round.

    Args:
        history     : list[dict] -- tracker.history
        figures_dir : str        -- Output directory for figures.
    """
    rounds      = _rounds(history)
    gd_loss     = [rec["gd_loss"] for rec in history]
    alpha_chg   = [rec["alpha_max_change"] for rec in history]

    # alpha_change is None for round 0 — skip it
    chg_rounds  = [r for r, v in zip(rounds, alpha_chg) if v is not None]
    chg_vals    = [v for v in alpha_chg if v is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rounds, gd_loss, marker="o", linewidth=1.5, markersize=3)
    axes[0].set_xlabel("GD round")
    axes[0].set_ylabel("GD loss")
    axes[0].set_title("GD loss per round")
    axes[0].grid(True, alpha=0.3)

    if chg_vals:
        axes[1].semilogy(chg_rounds, chg_vals, marker="o", linewidth=1.5,
                         markersize=3, color="darkorange")
        axes[1].set_xlabel("GD round")
        axes[1].set_ylabel("max |Δalpha|  (log scale)")
        axes[1].set_title("Alpha max change per round")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    _save(fig, os.path.join(figures_dir, "fig_gd_convergence.png"))


def plot_em_convergence(history: list, sample_names: list, figures_dir: str) -> None:
    """
    Figure 2: EM rounds per sample and EM convergence rate per GD round.

    Args:
        history      : list[dict]  -- tracker.history
        sample_names : list[str]   -- Sample names.
        figures_dir  : str         -- Output directory.
    """
    rounds = _rounds(history)
    S      = len(sample_names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: EM rounds per sample (one line per sample)
    for i, sname in enumerate(sample_names):
        em_r = [rec["em_rounds"][i] for rec in history]
        axes[0].plot(rounds, em_r, marker="o", linewidth=1.5, markersize=3,
                     label=sname)
    axes[0].set_xlabel("GD round")
    axes[0].set_ylabel("EM rounds")
    axes[0].set_title("EM rounds to convergence per sample")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: convergence rate across samples per GD round (fraction that converged)
    conv_rate = [sum(rec["em_converged"]) / S for rec in history]
    axes[1].plot(rounds, conv_rate, marker="s", linewidth=1.5, markersize=4,
                 color="green")
    axes[1].set_xlabel("GD round")
    axes[1].set_ylabel("Fraction converged")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].set_title("Fraction of samples where EM converged")
    axes[1].grid(True, alpha=0.3)

    _save(fig, os.path.join(figures_dir, "fig_em_convergence.png"))


def plot_inter_sample_corr(history: list, figures_dir: str) -> None:
    """
    Figure 3: Spearman and Pearson correlation between all theta pairs per round.

    Args:
        history     : list[dict] -- tracker.history
        figures_dir : str        -- Output directory.
    """
    rounds = _rounds(history)
    if not history[0]["inter_sample_corr"]:
        print("[plot_training] No inter-sample pairs to plot.")
        return

    pairs = list(history[0]["inter_sample_corr"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for pair in pairs:
        label = f"{pair[0]} ↔ {pair[1]}"
        sp = [rec["inter_sample_corr"][pair]["spearman"] for rec in history]
        pr = [rec["inter_sample_corr"][pair]["pearson"]  for rec in history]
        axes[0].plot(rounds, sp, marker="o", linewidth=1.5, markersize=3, label=label)
        axes[1].plot(rounds, pr, marker="o", linewidth=1.5, markersize=3, label=label)

    for ax, title in zip(axes, ["Spearman (theta_i vs theta_j)",
                                 "Pearson  (theta_i vs theta_j)"]):
        ax.set_xlabel("GD round")
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _save(fig, os.path.join(figures_dir, "fig_inter_sample_corr.png"))


def plot_theta_vs_alpha_corr(history: list, sample_names: list,
                              figures_dir: str) -> None:
    """
    Figure 4: Spearman and Pearson correlation between each theta_s and
    normalized alpha per GD round.

    Args:
        history      : list[dict] -- tracker.history
        sample_names : list[str]  -- Sample names.
        figures_dir  : str        -- Output directory.
    """
    rounds = _rounds(history)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for i, sname in enumerate(sample_names):
        sp = [rec["theta_vs_alpha_corr"][i]["spearman"] for rec in history]
        pr = [rec["theta_vs_alpha_corr"][i]["pearson"]  for rec in history]
        axes[0].plot(rounds, sp, marker="o", linewidth=1.5, markersize=3, label=sname)
        axes[1].plot(rounds, pr, marker="o", linewidth=1.5, markersize=3, label=sname)

    for ax, title in zip(axes, ["Spearman (theta_s vs alpha)",
                                 "Pearson  (theta_s vs alpha)"]):
        ax.set_xlabel("GD round")
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _save(fig, os.path.join(figures_dir, "fig_theta_vs_alpha_corr.png"))


def plot_alpha_stats(history: list, figures_dir: str) -> None:
    """
    Figure 5: Alpha sum and alpha entropy per GD round.

    Args:
        history     : list[dict] -- tracker.history
        figures_dir : str        -- Output directory.
    """
    rounds  = _rounds(history)
    a_sum   = [rec["alpha_sum"]     for rec in history]
    a_entr  = [rec["alpha_entropy"] for rec in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rounds, a_sum, marker="o", linewidth=1.5, markersize=3,
                 color="steelblue")
    axes[0].set_xlabel("GD round")
    axes[0].set_ylabel("sum(alpha)")
    axes[0].set_title("Alpha sum per round")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, a_entr, marker="o", linewidth=1.5, markersize=3,
                 color="tomato")
    axes[1].set_xlabel("GD round")
    axes[1].set_ylabel("Entropy H(alpha)")
    axes[1].set_title("Alpha entropy per round\n(higher = more uniform prior)")
    axes[1].grid(True, alpha=0.3)

    _save(fig, os.path.join(figures_dir, "fig_alpha_stats.png"))


def plot_nonzero_transcripts(history: list, sample_names: list,
                              figures_dir: str) -> None:
    """
    Figure 6: Number of non-zero transcripts per sample per GD round.

    Args:
        history      : list[dict] -- tracker.history
        sample_names : list[str]  -- Sample names.
        figures_dir  : str        -- Output directory.
    """
    rounds = _rounds(history)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, sname in enumerate(sample_names):
        nz = [rec["nonzero_per_sample"][i] for rec in history]
        ax.plot(rounds, nz, marker="o", linewidth=1.5, markersize=3, label=sname)

    ax.set_xlabel("GD round")
    ax.set_ylabel("Non-zero transcripts")
    ax.set_title("Non-zero transcripts per sample per round")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    _save(fig, os.path.join(figures_dir, "fig_nonzero_transcripts.png"))


# ============================================================
# Master function: generate all plots
# ============================================================

def generate_all_plots(tracker: TrainingTracker, figures_dir: str) -> None:
    """
    Generate all training diagnostic figures and save them to figures_dir.

    Called from main_multisample_joli.py after training completes, and also
    usable standalone via CLI.

    Args:
        tracker     : TrainingTracker -- Populated tracker from a completed run.
        figures_dir : str             -- Directory to write figure files.
    """
    os.makedirs(figures_dir, exist_ok=True)

    if not tracker.history:
        print("[plot_training] No history to plot (tracker is empty).")
        return

    history      = tracker.history
    sample_names = tracker.sample_names

    print(f"\n[plot_training] Generating {6} figures → {figures_dir}")

    plot_gd_convergence(history, figures_dir)
    plot_em_convergence(history, sample_names, figures_dir)
    plot_inter_sample_corr(history, figures_dir)
    plot_theta_vs_alpha_corr(history, sample_names, figures_dir)
    plot_alpha_stats(history, figures_dir)
    plot_nonzero_transcripts(history, sample_names, figures_dir)

    print(f"[plot_training] All figures saved to: {figures_dir}")


# ============================================================
# Standalone CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for standalone use.

    Returns:
        argparse.Namespace -- parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate training plots from a saved TrainingTracker pkl file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stats_path", required=True,
        help="Path to training_stats.pkl produced by main_multisample_joli.py."
    )
    parser.add_argument(
        "--figures_dir", default=None,
        help="Directory to save figures. Defaults to figures/ next to stats_path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    tracker = TrainingTracker.load(args.stats_path)
    print(f"[plot_training] Loaded tracker: "
          f"{len(tracker.history)} rounds, "
          f"samples={tracker.sample_names}")

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(args.stats_path), "figures"
    )
    generate_all_plots(tracker, figures_dir)
