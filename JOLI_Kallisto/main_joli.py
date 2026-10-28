"""
main_joli.py
============
Step 1.5 — JOLI EM entry point (Python only, no subprocess calls).

NOTE: This script does NOT run kallisto bus or bustools.
It assumes count.mtx, matrix.ec, and transcripts.txt already exist in --sample_dir.
To run the full pipeline (bustools + JOLI EM), use run_joli_kallisto.sh instead.

Takes a directory already containing bustools TCC output files:
  count.mtx, matrix.ec, transcripts.txt, run_info.json

Runs ONLY the JOLI EM pipeline (Step 4):
  Step 1.1  load_tcc.py     : parse bustools output -> TCCData
  Step 1.2  weights.py      : compute effective lengths and EC weights
  Step 1.3  em_algorithm.py : run JoliEM on equivalence classes
  Step 1.4  output_writer.py: write abundance.tsv in kallisto format

Use this script when:
  - The bustools TCC files already exist (cache hit in run_joli_kallisto.sh)
  - You want to set breakpoints in the EM logic
  - You want to re-run the EM with different settings without re-running bustools

Inputs:
  --sample_dir         : directory with count.mtx, matrix.ec, transcripts.txt
  --output_dir         : where to write abundance.tsv (default: same as sample_dir)
  --eff_len_mode       : "uniform" | "kallisto" | "fld"
                         "fld" (preferred for short reads): computes eff_lens in Python
                         from flens.txt (FLD histogram from kallisto bus --paired) +
                         transcript lengths from --transcriptome_fasta.
                         No kallisto quant call needed.
  --fld_path           : path to flens.txt (FLD histogram, 1000 values) written by
                         kallisto bus --paired. Required when --eff_len_mode fld.
  --transcriptome_fasta: path to transcriptome.fasta for parsing transcript lengths.
                         Required when --eff_len_mode fld.
  --max_em_rounds      : max EM iterations (default: 10000)
  --min_rounds         : min EM iterations before convergence check (default: 50)
  --em_type            : "plain" (Phase 1) | "MAP" (Phase 2) | "VI" (Phase 4)
  --convergence_mode   : "kallisto" (raw count threshold, matches lr-kallisto exactly) |
                         "joli"     (normalized theta threshold, faster, for MAP/VI)

Outputs:
  <output_dir>/abundance.tsv  : per-transcript quantification
  <output_dir>/runtime.txt    : wall-clock time for this run

Run:
  conda activate NanoCount_5
  python main_joli.py --sample_dir /path/to/kallisto_output/sample_stem/
"""

import argparse
import os
import pickle
import sys
import time

# Add core/ to path for library module imports

from joli.load_tcc import load_tcc_data, load_flens
from joli.weights import (compute_weights, load_fld_histogram,
                     load_fasta_transcript_lengths)
from joli.em_algorithm import JoliEM
from joli.output_writer import write_abundance


# ============================================================
# CONFIG — default values for all CLI arguments; edit here or pass as flags
# ============================================================
DEFAULT_SAMPLE_DIR        = "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/PacBio_data_fastq/PacBio/reads/long/downsampled/kallisto_output/toy"
DEFAULT_OUTPUT_DIR        = ""           # defaults to sample_dir if empty
DEFAULT_EFF_LEN_MODE      = "uniform"    # "uniform" | "kallisto" | "fld"
DEFAULT_FLD_PATH          = ""           # path to flens.txt (FLD histogram from kallisto bus --paired)
DEFAULT_TRANSCRIPTOME_FASTA = ""         # path to transcriptome.fasta (for "fld" mode)
DEFAULT_MAX_EM_ROUNDS = 10000
DEFAULT_MIN_ROUNDS    = 50
DEFAULT_EM_TYPE            = "plain"     # "plain" | "MAP" | "VI"
DEFAULT_CONVERGENCE_MODE   = "kallisto"  # "kallisto" (raw count threshold, matches LK)
                                         # "joli"     (normalized theta, faster, for MAP/VI)
SAVE_SNAPSHOTS             = False       # True = save theta snapshots every SNAPSHOT_INTERVAL
#                                        #   rounds to <output_dir>/theta_snapshots.pkl
#                                        #   Used by plot_convergence_animation.py
SNAPSHOT_INTERVAL          = 5          # save a snapshot every N EM rounds
EM_INCLUDE_SINGLE_TX       = True       # True  = single-tx counts included in M-step every round
#                                        #   (corrected behaviour — E-step uses total abundance)
#                                        # False = original kallisto behaviour
#                                        #   (single-tx added post-convergence only)
# ============================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace -- parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="JOLI EM: isoform quantification from kallisto/bustools TCC files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample_dir", default=DEFAULT_SAMPLE_DIR,
        help="Directory containing count.mtx, matrix.ec, transcripts.txt "
             "(output of bustools count)."
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to write abundance.tsv. Defaults to --sample_dir if empty."
    )
    parser.add_argument(
        "--eff_len_mode", default=DEFAULT_EFF_LEN_MODE,
        choices=["uniform", "kallisto", "fld"],
        help="Effective length mode. 'uniform'=all 1.0 (long reads); "
             "'fld'=Python FLD computation from flens.txt+FASTA (preferred for short reads); "
             "'kallisto'=from joli_efflen.txt (backward compat)."
    )
    parser.add_argument(
        "--fld_path", default=DEFAULT_FLD_PATH,
        help="Path to flens.txt (FLD histogram, 1000 values) from kallisto bus --paired. "
             "Required when --eff_len_mode fld."
    )
    parser.add_argument(
        "--transcriptome_fasta", default=DEFAULT_TRANSCRIPTOME_FASTA,
        help="Path to transcriptome.fasta for parsing transcript lengths. "
             "Required when --eff_len_mode fld."
    )
    parser.add_argument(
        "--max_em_rounds", type=int, default=DEFAULT_MAX_EM_ROUNDS,
        help="Maximum EM iterations."
    )
    parser.add_argument(
        "--min_rounds", type=int, default=DEFAULT_MIN_ROUNDS,
        help="Minimum EM iterations before checking convergence (matches kallisto default)."
    )
    parser.add_argument(
        "--em_type", default=DEFAULT_EM_TYPE,
        choices=["plain", "MAP", "VI"],
        help="EM mode. 'plain'=standard EM (Phase 1); 'MAP'=Dirichlet MAP (Phase 2); "
             "'VI'=variational inference (Phase 4)."
    )
    parser.add_argument(
        "--convergence_mode", default=DEFAULT_CONVERGENCE_MODE,
        choices=["kallisto", "joli"],
        help="Convergence criterion. 'kallisto'=raw expected count threshold (matches "
             "lr-kallisto exactly, recommended for plain EM comparison); "
             "'joli'=normalized theta threshold (faster, recommended for MAP/VI)."
    )
    parser.add_argument(
        "--save_snapshots", default=str(SAVE_SNAPSHOTS).lower(),
        choices=["true", "false"],
        help="Save theta snapshots every --snapshot_interval rounds to theta_snapshots.pkl."
    )
    parser.add_argument(
        "--snapshot_interval", type=int, default=SNAPSHOT_INTERVAL,
        help="Save a theta snapshot every N EM rounds (only used when --save_snapshots true)."
    )
    parser.add_argument(
        "--em_include_single_tx", default=str(EM_INCLUDE_SINGLE_TX).lower(),
        choices=["true", "false"],
        help="Include single-tx read counts in the EM M-step every round. "
             "true (default) = corrected behaviour; false = original kallisto behaviour."
    )
    return parser.parse_args()


def save_runtime(output_dir: str, elapsed: float) -> None:
    """
    Write elapsed wall-clock time to runtime.txt.

    Args:
        output_dir : str   -- Output directory.
        elapsed    : float -- Elapsed seconds.
    """
    path = os.path.join(output_dir, "runtime.txt")
    with open(path, "w") as fh:
        fh.write(f"{elapsed:.2f} seconds\n")
    print(f"[main_joli] Runtime saved: {elapsed:.2f}s -> {path}")


def main() -> None:
    """
    JOLI EM pipeline: load TCC -> compute weights -> run EM -> write abundance.tsv.
    """
    run_start = time.time()
    args = parse_args()

    # Resolve paths
    sample_dir = os.path.abspath(args.sample_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else sample_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("JOLI-Kallisto: main_joli.py")
    print("=" * 60)
    print(f"  sample_dir          : {sample_dir}")
    print(f"  output_dir          : {output_dir}")
    print(f"  eff_len_mode        : {args.eff_len_mode}")
    if args.eff_len_mode == "fld":
        print(f"  fld_path            : {args.fld_path}")
        print(f"  transcriptome_fasta : {args.transcriptome_fasta}")
    print(f"  em_type             : {args.em_type}")
    print(f"  max_em_rounds    : {args.max_em_rounds}")
    print(f"  min_rounds       : {args.min_rounds}")
    print(f"  convergence_mode : {args.convergence_mode}")
    print(f"  save_snapshots   : {args.save_snapshots}")
    print(f"  snapshot_interval: {args.snapshot_interval}")
    print(f"  em_include_single_tx: {args.em_include_single_tx}")
    print("=" * 60)

    # Resolve boolean flags (CLI passes string "true"/"false")
    save_snapshots       = args.save_snapshots.lower()       == "true"
    em_include_single_tx = args.em_include_single_tx.lower() == "true"

    # --- Step 1.1: Load TCC data ---
    tcc_data = load_tcc_data(sample_dir)

    # --- Step 1.2: Compute weights ---
    if args.eff_len_mode == "fld":
        # Python FLD path: compute eff_lens from FLD histogram + FASTA transcript lengths.
        # No kallisto quant call needed — flens.txt is produced by kallisto bus --paired.
        if not args.fld_path:
            raise ValueError("--eff_len_mode fld requires --fld_path (path to flens.txt).")
        if not args.transcriptome_fasta:
            raise ValueError(
                "--eff_len_mode fld requires --transcriptome_fasta (path to FASTA)."
            )
        fld = load_fld_histogram(args.fld_path)
        tx_lengths = load_fasta_transcript_lengths(
            args.transcriptome_fasta, tcc_data.transcript_names
        )
        weight_data = compute_weights(
            tcc_data, transcript_lengths=tx_lengths, fld=fld, mode="fld"
        )
    elif args.eff_len_mode == "kallisto":
        # Backward-compat: load pre-computed eff_lens from joli_efflen.txt.
        flens = load_flens(sample_dir, n_transcripts=len(tcc_data.transcript_names))
        weight_data = compute_weights(tcc_data, flens=flens, mode="kallisto")
    else:
        weight_data = compute_weights(tcc_data, mode=args.eff_len_mode)

    # --- Step 1.3: Run EM ---
    if args.em_type != "plain":
        # TODO Phase 2 (MAP) and Phase 4 (VI)
        raise NotImplementedError(
            f"em_type='{args.em_type}' is not yet implemented. "
            "Use --em_type plain for Phase 1."
        )

    em = JoliEM(tcc_data, weight_data, em_include_single_tx=em_include_single_tx)
    em_result = em.run(
        max_em_rounds     = args.max_em_rounds,
        min_rounds        = args.min_rounds,
        convergence_mode  = args.convergence_mode,
        snapshot_interval = args.snapshot_interval if save_snapshots else 0,
    )

    print(f"\n[main_joli] EM finished: "
          f"rounds={em_result.n_rounds}, converged={em_result.converged}, "
          f"nonzero_tx={int((em_result.theta_unnorm > 0).sum())}")

    # --- Save theta snapshots (if enabled) ---
    if save_snapshots and em_result.snapshots:
        snap_path = os.path.join(output_dir, "theta_snapshots.pkl")
        with open(snap_path, "wb") as fh:
            pickle.dump({
                "transcript_names": tcc_data.transcript_names,
                "snapshots":        em_result.snapshots,   # list of (round_num, theta)
            }, fh)
        print(f"[main_joli] Snapshots saved ({len(em_result.snapshots)} frames): {snap_path}")

    # --- Step 1.4: Write abundance.tsv ---
    output_path = os.path.join(output_dir, "abundance.tsv")
    summary = write_abundance(
        theta_unnorm=em_result.theta_unnorm,
        eff_lens=weight_data.eff_lens,
        transcript_names=tcc_data.transcript_names,
        output_path=output_path,
    )

    # --- Save runtime ---
    elapsed = time.time() - run_start
    save_runtime(output_dir, elapsed)

    print(f"\n[main_joli] Done.")
    print(f"  abundance.tsv   : {output_path}")
    print(f"  non-zero TPM    : {summary['n_nonzero']} / {summary['n_transcripts']}")
    print(f"  total TPM       : {summary['total_tpm']:.2f}")
    print(f"  total est_counts: {summary['total_est_counts']:.2f}")
    print(f"  total elapsed   : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
