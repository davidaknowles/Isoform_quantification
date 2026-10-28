"""
main_multisample_joli.py
========================
Entry point for JOLI-Kallisto Phase 2: multi-sample MAP EM.

Assumes bustools TCC output files already exist in each sample directory
(count.mtx, matrix.ec, transcripts.txt, run_info.json). To run the full
upstream pipeline (kallisto bus → bustools sort/count) first, use
scripts/run_multisample_joli.sh instead.

Pipeline:
  For each sample:
    Step 1.1  load_tcc.py     : parse bustools output → TCCData
    Step 1.2  weights.py      : compute effective lengths and EC weights
  Shared:
    Step 2.1  multi_sample_em.py   : outer GD loop + per-sample MAP EM
    Step 2.2  dirichlet_optimizer.py : Adam update of shared alpha
  Output:
    Per-sample abundance.tsv, alpha_final.npy, gd_loss_history.pkl

Sample input — two options (set ONE in CONFIG, leave the other blank/empty):
  Option A — SAMPLE_DIRS: explicit list of bustools output directories.
  Option B — SAMPLES_FOLDER: parent folder; code auto-discovers all subdirs.

CLI args (all optional — override CONFIG when provided):
  --sample_dirs DIR [DIR ...]          Explicit list of bustools output directories.
  --results_base PATH                  Base directory for timestamped output folder.
  --eff_len_mode STR                   "uniform" | "kallisto" | "fld" (fallback for all samples).
  --eff_len_modes MODE [MODE ...]      Per-sample mode list (overrides --eff_len_mode).
  --transcriptome_fasta PATH           FASTA file required when any sample uses "fld" mode.
  --em_include_single_tx_list T/F ...  Per-sample list of "true"/"false" (overrides --em_include_single_tx).
  --convergence_mode STR               "joli" | "kallisto"
  --max_em_rounds INT                  Max inner EM rounds per sample per GD round.
  --min_em_rounds INT                  Min inner EM rounds before convergence check.
  --max_gd_rounds INT                  Max outer GD iterations.
  --gd_lr FLOAT                        Adam learning rate for shared alpha.
  --alpha_initial FLOAT                Initial Dirichlet concentration.
  --gd_convergence_tol FLOAT           Stop outer loop when |loss_change| < tol.
  --gd_steps_per_round INT             Adam steps per outer GD round.
  --min_read_support FLOAT             Fix A: min n[t] to apply Dirichlet prior (0.0 = off).
  --run_dir_output PATH                Write the created run_dir path to this file (for bash).

Inputs:
  SAMPLE_DIRS or SAMPLES_FOLDER (CONFIG) or --sample_dirs (CLI)

Outputs (per experiment run):
  <RESULTS_BASE>/exprmnt_{timestamp}/
    experiment_description.log
    running.log
    runtime.txt
    code_snapshot/
    alpha_final.npy
    gd_loss_history.pkl
    <sample_name>/
      abundance.tsv

Run (standalone):
  conda activate NanoCount_5
  python main_multisample_joli.py

Run (from bash pipeline — passes sample dirs dynamically):
  python main_multisample_joli.py --sample_dirs /path/s1 /path/s2 /path/s3
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


from joli.multi_sample_em import MultiSampleJoliEM


# ============================================================
# CONFIG — edit everything here; do not touch pipeline logic below
# ============================================================

# --- Sample input: use ONE of the two options ---
# Option A: explicit list of sample dirs
SAMPLE_DIRS = [   
    "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/PacBio_data_fastq/PacBio/reads/long/downsampled/kallisto_output/ds_52_furtherDownsampled",
    "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/PacBio_data_fastq/PacBio/reads/long/downsampled/kallisto_output/ds_52_furtherDownsampled",
    # "/path/to/kallisto_output/sample_1/",
    # "/path/to/kallisto_output/sample_2/",
]
SAMPLES_FOLDER = ""                     # Option B: parent folder containing all sample subdirs
#                                       #   leave "" to use SAMPLE_DIRS instead

RESULTS_BASE        = "/gpfs/commons/home/atalukder/RNA_Splicing/files/results"
READ_TYPE           = "long"            # informational only: "long" | "short" | "mixed"
#                                       # the actual per-sample read type is handled by the
#                                       # bash pipeline (run_multisample_joli.sh)
EFF_LEN_MODE        = "uniform"         # fallback for all samples: "uniform" | "fld" | "kallisto"
EFF_LEN_MODES       = []               # per-sample list overrides EFF_LEN_MODE;
#                                       #   e.g. ["uniform", "fld", "fld"] for LR + 2×SR samples
#                                       #   leave [] to use EFF_LEN_MODE for all samples
TRANSCRIPTOME_FASTA = ""               # path to transcriptome.fasta — required when any sample
#                                       #   uses eff_len_mode="fld" (short reads)
CONVERGENCE_MODE    = "joli"            # "joli" (recommended for MAP) | "kallisto"
MAX_EM_ROUNDS       = 10000             # max inner EM rounds per sample per GD round
MIN_EM_ROUNDS       = 1                # min inner EM rounds before convergence check
MAX_GD_ROUNDS       = 500              # max outer GD iterations
GD_LR               = 0.01             # Adam learning rate for shared alpha
ALPHA_INITIAL       = 1.0              # initial Dirichlet concentration (uniform prior)
GD_CONVERGENCE_TOL  = 1e-6             # stop outer loop when |gd_loss_change| < this
GD_STEPS_PER_ROUND  = 10               # Adam steps per outer GD round
MIN_READ_SUPPORT    = 0.0              # Fix A flag: min expected reads n[t] to apply prior
#                                      #   0.0 = disabled (prior always applied)
#                                      #   try 0.1 to gate prior on real read support
SAVE_SNAPSHOTS      = False             # True = save alpha + theta snapshots every
#                                      #   SNAPSHOT_INTERVAL rounds to snapshots.pkl
#                                      #   Used by plot_convergence_animation.py
SNAPSHOT_INTERVAL   = 5               # save every N rounds (only when SAVE_SNAPSHOTS=True)
EM_INCLUDE_SINGLE_TX = True           # True  = single-tx counts included in EM M-step every round
#                                      #   (corrected behaviour — E-step uses total abundance)
#                                      # False = original kallisto behaviour
#                                      #   (single-tx added post-convergence only)
EM_INCLUDE_SINGLE_TX_LIST = []        # per-sample list overrides EM_INCLUDE_SINGLE_TX;
#                                      #   e.g. [False, True, True] for LR + 2×SR samples
#                                      #   leave [] to use EM_INCLUDE_SINGLE_TX for all samples
GD_INCLUDE_SINGLE_TX = False          # False (default) = single-tx excluded from theta passed to GD
#                                      #   GD gradient driven only by multi-tx (ambiguous) ECs —
#                                      #   where cross-sample sharing resolves ambiguity
#                                      # True  = old behaviour (single-tx baked into GD theta)
LOOP_MODE           = "gd_wrapper"     # Training loop structure:
#                                      #   "gd_wrapper" — GD outer loop, EM to convergence
#                                      #                  per round (original JOLI behaviour)
#                                      #   "em_wrapper" — EM convergence drives outer loop,
#                                      #                  1 EM step + GD_STEPS_PER_ROUND Adam
#                                      #                  steps per iteration (AT_code style)

EXPERIMENT_COMMENT  = ""               # free-text note saved in experiment_description.log

# ============================================================


# ============================================================
# CLI argument parsing (overrides CONFIG when provided)
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse optional CLI arguments that override CONFIG values.

    All arguments are optional. When not provided, the corresponding
    CONFIG variable is used unchanged. This allows the script to be
    called both interactively (edit CONFIG, run with no args) and
    from a bash pipeline (pass --sample_dirs and other settings as flags).

    Returns:
        argparse.Namespace -- parsed arguments (unset fields are None).
    """
    parser = argparse.ArgumentParser(
        description="JOLI multi-sample MAP EM: joint Dirichlet prior across samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample_dirs", nargs="+", default=None,
        help="Explicit list of bustools output directories (overrides CONFIG SAMPLE_DIRS)."
    )
    parser.add_argument(
        "--sample_names", nargs="+", default=None,
        help="Human-readable names for each sample (must match --sample_dirs order). "
             "If omitted, names are derived from the directory basename."
    )
    parser.add_argument(
        "--results_base", default=None,
        help="Base directory for timestamped output folder (overrides CONFIG RESULTS_BASE)."
    )
    parser.add_argument(
        "--eff_len_mode", default=None, choices=["uniform", "kallisto", "fld"],
        help="Effective length mode (fallback applied to all samples)."
    )
    parser.add_argument(
        "--eff_len_modes", nargs="+", default=None,
        help="Per-sample effective length modes (e.g. uniform fld fld). "
             "Overrides --eff_len_mode when provided. Must match number of samples."
    )
    parser.add_argument(
        "--transcriptome_fasta", default=None,
        help="Path to transcriptome FASTA file. Required when any sample uses mode='fld'."
    )
    parser.add_argument(
        "--em_include_single_tx_list", nargs="+", default=None,
        metavar="true|false",
        help="Per-sample em_include_single_tx flags (e.g. false true true). "
             "Overrides --em_include_single_tx. Must match number of samples."
    )
    parser.add_argument(
        "--run_dir_output", default=None,
        help="Write the created run_dir path to this file. Used by bash pipeline to "
             "capture the experiment folder name without parsing stdout."
    )
    parser.add_argument(
        "--run_dir", default=None,
        help="Exact output directory to use (overrides --results_base and the internal "
             "timestamp generation). Lets a batch submitter (e.g. submit_JKMS_slurm.py) "
             "decide each run's exprmnt_* path up front, so concurrent jobs launched by "
             "the same batch never collide on an independently-generated timestamp. "
             "When omitted (manual/standalone runs), the timestamp is generated internally "
             "as before."
    )
    parser.add_argument(
        "--convergence_mode", default=None, choices=["joli", "kallisto"],
        help="EM convergence criterion."
    )
    parser.add_argument(
        "--max_em_rounds", type=int, default=None,
        help="Max inner EM rounds per sample per GD round."
    )
    parser.add_argument(
        "--min_em_rounds", type=int, default=None,
        help="Min inner EM rounds before convergence check."
    )
    parser.add_argument(
        "--max_gd_rounds", type=int, default=None,
        help="Max outer GD iterations."
    )
    parser.add_argument(
        "--gd_lr", type=float, default=None,
        help="Adam learning rate for shared alpha."
    )
    parser.add_argument(
        "--alpha_initial", type=float, default=None,
        help="Initial Dirichlet concentration (uniform prior)."
    )
    parser.add_argument(
        "--gd_convergence_tol", type=float, default=None,
        help="Stop outer loop when |loss_change| < tol."
    )
    parser.add_argument(
        "--gd_steps_per_round", type=int, default=None,
        help="Adam steps per outer GD round."
    )
    parser.add_argument(
        "--min_read_support", type=float, default=None,
        help="Fix A flag: min expected reads n[t] required to apply Dirichlet prior. "
             "0.0 = disabled (prior always applied). Try 0.1 to reduce FP leakage."
    )
    parser.add_argument(
        "--save_snapshots", default=None, choices=["true", "false"],
        help="Save alpha+theta snapshots every --snapshot_interval rounds to snapshots.pkl."
    )
    parser.add_argument(
        "--snapshot_interval", type=int, default=None,
        help="Save a snapshot every this many rounds (only when --save_snapshots=true)."
    )
    parser.add_argument(
        "--loop_mode", default=None, choices=["gd_wrapper", "em_wrapper"],
        help="Training loop structure. "
             "'gd_wrapper': GD outer loop, EM to convergence per round (default). "
             "'em_wrapper': EM convergence drives outer loop, 1 EM step + N GD steps "
             "per iteration (AT_code style)."
    )
    parser.add_argument(
        "--em_include_single_tx", default=None, choices=["true", "false"],
        help="Include single-tx counts in the EM M-step every round. "
             "true (default) = corrected behaviour; false = original kallisto behaviour."
    )
    parser.add_argument(
        "--gd_include_single_tx", default=None, choices=["true", "false"],
        help="Include single-tx counts in the theta passed to the GD optimizer. "
             "false (default) = GD driven only by multi-tx ECs (recommended); "
             "true = old behaviour (single-tx baked into GD theta)."
    )
    parser.add_argument(
        "--experiment_comment", default=None,
        help="Free-text description of this run, saved in experiment_description.log."
    )
    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def resolve_sample_dirs() -> list:
    """
    Resolve the list of sample directories from CONFIG.

    Uses SAMPLES_FOLDER (auto-discover subdirs) if set, otherwise SAMPLE_DIRS.

    Returns:
        list[str] -- sorted list of absolute sample directory paths.
    """
    if SAMPLES_FOLDER:
        base = os.path.abspath(SAMPLES_FOLDER)
        dirs = sorted([
            os.path.join(base, d)
            for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ])
        print(f"[main_multisample] Auto-discovered {len(dirs)} samples in {base}")
        for d in dirs:
            print(f"  {os.path.basename(d)}")
        return dirs
    else:
        return [os.path.abspath(d) for d in SAMPLE_DIRS]


def create_run_dir() -> str:
    """
    Create and return a timestamped results directory.

    Returns:
        str -- path to the newly created run directory.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    run_dir   = os.path.join(RESULTS_BASE, f"exprmnt_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_experiment_description(run_dir: str, sample_dirs: list,
                                eff_len_mode: str, convergence_mode: str,
                                max_em_rounds: int, min_em_rounds: int,
                                max_gd_rounds: int, gd_lr: float,
                                alpha_initial: float, gd_convergence_tol: float,
                                gd_steps_per_round: int,
                                min_read_support: float,
                                loop_mode: str,
                                em_include_single_tx: bool,
                                gd_include_single_tx: bool,
                                experiment_comment: str) -> None:
    """
    Write experiment_description.log with effective runtime values and sample list.

    Args:
        run_dir              : str       -- Timestamped result directory.
        sample_dirs          : list[str] -- Resolved sample directories.
        eff_len_mode         : str       -- Effective length mode used.
        convergence_mode     : str       -- Convergence criterion used.
        max_em_rounds        : int       -- Max inner EM rounds.
        min_em_rounds        : int       -- Min inner EM rounds.
        max_gd_rounds        : int       -- Max outer GD iterations.
        gd_lr                : float     -- Adam learning rate.
        alpha_initial        : float     -- Initial Dirichlet concentration.
        gd_convergence_tol   : float     -- GD convergence tolerance.
        gd_steps_per_round   : int       -- Adam steps per GD round.
        min_read_support     : float     -- Fix A flag (0.0 = disabled).
        em_include_single_tx : bool      -- Single-tx in EM M-step flag.
        gd_include_single_tx : bool      -- Single-tx in GD theta flag.
        experiment_comment   : str       -- Free-text description of this run.
    """
    lines = [
        f"Script: {os.path.abspath(__file__)}",
        f"Timestamp: {datetime.now().isoformat()}",
        "",
    ]
    if experiment_comment:
        lines += [
            "=== EXPERIMENT COMMENT ===",
            experiment_comment,
            "",
        ]
    lines += [
        "=== CONFIG (effective values — CLI overrides CONFIG when provided) ===",
        f"READ_TYPE:          {READ_TYPE}",
        f"EFF_LEN_MODE:       {eff_len_mode}",
        f"CONVERGENCE_MODE:   {convergence_mode}",
        f"MAX_EM_ROUNDS:      {max_em_rounds}",
        f"MIN_EM_ROUNDS:      {min_em_rounds}",
        f"MAX_GD_ROUNDS:      {max_gd_rounds}",
        f"GD_LR:              {gd_lr}",
        f"ALPHA_INITIAL:      {alpha_initial}",
        f"GD_CONVERGENCE_TOL: {gd_convergence_tol}",
        f"GD_STEPS_PER_ROUND: {gd_steps_per_round}",
        f"MIN_READ_SUPPORT:   {min_read_support}",
        f"LOOP_MODE:           {loop_mode}",
        f"EM_INCLUDE_SINGLE_TX:{em_include_single_tx}",
        f"GD_INCLUDE_SINGLE_TX:{gd_include_single_tx}",
        "",
        "=== SAMPLES ===",
    ]
    for s in sample_dirs:
        lines.append(f"  {s}")

    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[main_multisample] Experiment description saved: {path}")


def save_runtime(run_dir: str, elapsed: float) -> None:
    """
    Write elapsed wall-clock time to runtime.txt.

    Args:
        run_dir : str   -- Timestamped result directory.
        elapsed : float -- Elapsed seconds.
    """
    path = os.path.join(run_dir, "runtime.txt")
    with open(path, "w") as fh:
        fh.write(f"{elapsed:.2f} seconds\n")
    print(f"[main_multisample] Runtime: {elapsed:.2f}s → {path}")


def save_code_snapshot(run_dir: str) -> None:
    """
    Copy all source files (.py, .sh, .txt, .yml, .yaml) from JOLI_Kallisto/
    into run_dir/code_snapshot/.

    Args:
        run_dir : str -- Timestamped result directory.
    """
    snapshot_dir = os.path.join(run_dir, "code_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    exts    = {".py", ".sh", ".txt", ".yml", ".yaml"}

    copied = 0
    for fpath in Path(src_dir).rglob("*"):
        if fpath.suffix in exts and fpath.is_file():
            shutil.copy2(str(fpath), os.path.join(snapshot_dir, fpath.name))
            copied += 1
    print(f"[main_multisample] Code snapshot saved ({copied} files): {snapshot_dir}")


def setup_logging(run_dir: str):
    """
    Redirect stdout to both terminal and running.log.

    Args:
        run_dir : str -- Timestamped result directory.

    Returns:
        file handle for running.log (caller should close at end).
    """
    log_path = os.path.join(run_dir, "running.log")
    log_fh   = open(log_path, "w")
    log_fh.write(f"Script: {os.path.abspath(__file__)}\n")
    log_fh.flush()

    class Tee:
        """Write to both terminal and log file."""
        def __init__(self, *streams):
            self.streams = streams
        def write(self, msg):
            for s in self.streams:
                s.write(msg)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log_fh)
    return log_fh


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Multi-sample JOLI MAP EM pipeline:
      merge CLI args over CONFIG → resolve samples → create run dir →
      run MultiSampleJoliEM → write outputs.

    CLI args override CONFIG when provided; CONFIG is used otherwise.
    """
    run_start = time.time()

    # --- Merge CLI args over CONFIG ---
    # Parse once; None means "not provided" so CONFIG value is kept.
    args = parse_args()

    # Effective values (CLI wins over CONFIG when CLI arg is not None)
    eff_len_mode       = args.eff_len_mode       or EFF_LEN_MODE
    # Per-sample modes: CLI list > CONFIG list > None (falls back to eff_len_mode for all)
    eff_len_modes      = (
        args.eff_len_modes if args.eff_len_modes is not None
        else (EFF_LEN_MODES if EFF_LEN_MODES else None)
    )
    transcriptome_fasta = (
        args.transcriptome_fasta if args.transcriptome_fasta is not None
        else (TRANSCRIPTOME_FASTA if TRANSCRIPTOME_FASTA else None)
    )
    # Per-sample em_include_single_tx list: strings "true"/"false" → booleans
    _em_list_raw = (
        args.em_include_single_tx_list if args.em_include_single_tx_list is not None
        else (EM_INCLUDE_SINGLE_TX_LIST if EM_INCLUDE_SINGLE_TX_LIST else None)
    )
    em_include_single_tx_list = (
        [v.lower() == "true" for v in _em_list_raw] if _em_list_raw else None
    )
    convergence_mode   = args.convergence_mode   or CONVERGENCE_MODE
    max_em_rounds      = args.max_em_rounds      if args.max_em_rounds      is not None else MAX_EM_ROUNDS
    min_em_rounds      = args.min_em_rounds      if args.min_em_rounds      is not None else MIN_EM_ROUNDS
    max_gd_rounds      = args.max_gd_rounds      if args.max_gd_rounds      is not None else MAX_GD_ROUNDS
    gd_lr              = args.gd_lr              if args.gd_lr              is not None else GD_LR
    alpha_initial      = args.alpha_initial      if args.alpha_initial      is not None else ALPHA_INITIAL
    gd_convergence_tol = args.gd_convergence_tol if args.gd_convergence_tol is not None else GD_CONVERGENCE_TOL
    gd_steps_per_round = args.gd_steps_per_round if args.gd_steps_per_round is not None else GD_STEPS_PER_ROUND
    min_read_support   = args.min_read_support   if args.min_read_support   is not None else MIN_READ_SUPPORT
    loop_mode          = args.loop_mode          or LOOP_MODE
    save_snapshots     = (args.save_snapshots == "true") if args.save_snapshots is not None else SAVE_SNAPSHOTS
    snapshot_interval  = args.snapshot_interval if args.snapshot_interval is not None else SNAPSHOT_INTERVAL
    em_include_single_tx = (args.em_include_single_tx == "true") if args.em_include_single_tx is not None else EM_INCLUDE_SINGLE_TX
    gd_include_single_tx = (args.gd_include_single_tx == "true") if args.gd_include_single_tx is not None else GD_INCLUDE_SINGLE_TX
    results_base       = args.results_base or RESULTS_BASE
    experiment_comment = args.experiment_comment if args.experiment_comment is not None else EXPERIMENT_COMMENT

    # --- Resolve sample directories ---
    # CLI --sample_dirs takes priority over CONFIG SAMPLE_DIRS / SAMPLES_FOLDER.
    if args.sample_dirs is not None:
        sample_dirs = [os.path.abspath(d) for d in args.sample_dirs]
    else:
        sample_dirs = resolve_sample_dirs()

    if len(sample_dirs) < 2:
        print(f"ERROR: Need at least 2 samples, found {len(sample_dirs)}. "
              f"Check SAMPLE_DIRS or SAMPLES_FOLDER in CONFIG, or pass --sample_dirs.")
        sys.exit(1)

    # --- Create run directory and start logging ---
    # --run_dir (set by a batch submitter) takes priority so concurrent jobs
    # launched together never race on independently-generated timestamps.
    # Manual/standalone runs (no --run_dir) keep the original behaviour.
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        run_dir   = os.path.join(results_base, f"exprmnt_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
    log_fh  = setup_logging(run_dir)

    print("=" * 60)
    print("JOLI-Kallisto Phase 2: main_multisample_joli.py")
    print("=" * 60)
    print(f"  run_dir          : {run_dir}")
    print(f"  n_samples        : {len(sample_dirs)}")
    print(f"  eff_len_mode     : {eff_len_mode}  (fallback)")
    print(f"  eff_len_modes    : {eff_len_modes or '(all use fallback)'}")
    print(f"  transcriptome_fasta : {transcriptome_fasta or '(not set)'}")
    print(f"  convergence_mode : {convergence_mode}")
    print(f"  max_em_rounds    : {max_em_rounds}")
    print(f"  min_em_rounds    : {min_em_rounds}")
    print(f"  max_gd_rounds    : {max_gd_rounds}")
    print(f"  gd_lr            : {gd_lr}")
    print(f"  alpha_initial    : {alpha_initial}")
    print(f"  gd_conv_tol      : {gd_convergence_tol}")
    print(f"  gd_steps/em_round   : {gd_steps_per_round}")
    print(f"  min_read_support    : {min_read_support}  (Fix A: 0.0 = disabled)")
    print(f"  loop_mode           : {loop_mode}")
    print(f"  save_snapshots         : {save_snapshots}  (interval={snapshot_interval})")
    print(f"  em_include_single_tx      : {em_include_single_tx}  (fallback)")
    print(f"  em_include_single_tx_list : {em_include_single_tx_list or '(all use fallback)'}")
    print(f"  gd_include_single_tx      : {gd_include_single_tx}")
    if experiment_comment:
        print(f"  comment             : {experiment_comment}")
    print("=" * 60)

    # Save experiment description and code snapshot before running
    # Write run_dir path to file if requested (used by bash pipeline to capture it)
    if args.run_dir_output:
        with open(args.run_dir_output, "w") as _f:
            _f.write(run_dir + "\n")
        print(f"[main_multisample] run_dir written to: {args.run_dir_output}")

    save_experiment_description(
        run_dir, sample_dirs,
        eff_len_mode        = eff_len_mode,
        convergence_mode    = convergence_mode,
        max_em_rounds       = max_em_rounds,
        min_em_rounds       = min_em_rounds,
        max_gd_rounds       = max_gd_rounds,
        gd_lr               = gd_lr,
        alpha_initial       = alpha_initial,
        gd_convergence_tol  = gd_convergence_tol,
        gd_steps_per_round  = gd_steps_per_round,
        min_read_support    = min_read_support,
        loop_mode            = loop_mode,
        em_include_single_tx = em_include_single_tx,
        gd_include_single_tx = gd_include_single_tx,
        experiment_comment   = experiment_comment,
    )
    save_code_snapshot(run_dir)

    # --- Run multi-sample MAP EM ---
    ms = MultiSampleJoliEM(
        sample_dirs              = sample_dirs,
        sample_names             = args.sample_names,   # None → derived from dir basename
        eff_len_mode             = eff_len_mode,
        eff_len_modes            = eff_len_modes,
        transcriptome_fasta      = transcriptome_fasta,
        convergence_mode         = convergence_mode,
        max_em_rounds            = max_em_rounds,
        min_em_rounds            = min_em_rounds,
        max_gd_rounds            = max_gd_rounds,
        gd_lr                    = gd_lr,
        alpha_initial            = alpha_initial,
        gd_convergence_tol       = gd_convergence_tol,
        gd_steps_per_round       = gd_steps_per_round,
        min_read_support         = min_read_support,
        loop_mode                = loop_mode,
        save_snapshots           = save_snapshots,
        snapshot_interval        = snapshot_interval,
        em_include_single_tx     = em_include_single_tx,
        em_include_single_tx_list= em_include_single_tx_list,
        gd_include_single_tx     = gd_include_single_tx,
    )

    results = ms.run()

    # --- Write outputs ---
    ms.write_results(run_dir, results)

    # --- Save runtime ---
    elapsed = time.time() - run_start
    save_runtime(run_dir, elapsed)

    print(f"\n[main_multisample] Done.")
    print(f"  run_dir      : {run_dir}")
    print(f"  n_gd_rounds  : {results['n_gd_rounds']}")
    print(f"  gd_converged : {results['converged']}")
    print(f"  elapsed      : {elapsed:.2f}s")

    log_fh.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
