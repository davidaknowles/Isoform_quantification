"""
Joint SR+LR quantification on the LRGASP-style simulation (two samples, each
with PacBio long reads and Illumina paired-end short reads simulated from the
same generating abundances). Fits LR only, SR only and SR+LR per sample and
scores TPM against the generating truth.

Usage: python analyses/sim_joint.py [out.tsv]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.evaluate import fraction_metrics, read_t2g, read_tpm, transcript_metrics
from joli.joint import JointEM, library_from_tcc, molar_to_tpm
from joli.load_tcc import load_tcc_data
from joli.weights import (calc_eff_lens_from_fld, compute_mean_fl_trunc,
                          load_fasta_transcript_lengths, load_fld_histogram)

SIM = Path("/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/sim_real_data")
LK = Path("/gpfs/commons/home/atalukder/RNA_Splicing/data/Shree_stuff/SOTA/lr-kallisto")
SAMPLES = {"sim1": ("ds_100_num1_aln_01", "1"), "sim2": ("ds_100_num1_aln_21", "2")}


def quiet(f, *args, **kw):
    """Run f with stdout silenced (the loaders print progress)."""
    with open(os.devnull, "w") as dn:
        old, sys.stdout = sys.stdout, dn
        try:
            return f(*args, **kw)
        finally:
            sys.stdout = old


def main(out_path=None):
    t2g = read_t2g(LK / "t2g.txt")
    tx_len = None
    rows = []
    for sample, (stem, k) in SAMPLES.items():
        lr_tcc = quiet(load_tcc_data, SIM / "kallisto_output" / f"{stem}_long")
        sr_dir = SIM / "kallisto_output" / f"{stem}_short_1"
        sr_tcc = quiet(load_tcc_data, sr_dir)
        if tx_len is None:
            tx_len = quiet(load_fasta_transcript_lengths, LK / "transcriptome.fasta", sr_tcc.transcript_names)
        fld = quiet(load_fld_histogram, sr_dir / "flens.txt")
        eff = quiet(calc_eff_lens_from_fld, tx_len, quiet(compute_mean_fl_trunc, fld))
        lr = library_from_tcc("lr", lr_tcc, "long")
        sr = library_from_tcc("sr", sr_tcc, "short", eff_lens=eff)
        truth = read_tpm(SIM / "ground_truths" / f"sample{k}_gt.tsv")
        for fit, libs in {"LR": [lr], "SR": [sr], "SR+LR": [sr, lr]}.items():
            t0 = time.time()
            res = JointEM(libs, units=[0] * len(libs)).run(max_iter=5000)
            tpm = pd.Series(molar_to_tpm(res.theta[0]), index=lr_tcc.transcript_names)
            row = {"sample": sample, "fit": fit, "n_iter": res.n_iter, "converged": res.converged,
                   "seconds": round(time.time() - t0, 1)}
            row.update(transcript_metrics(tpm, truth))
            row.update(fraction_metrics(tpm, truth, t2g))
            rows.append(row)
            print(pd.DataFrame([row]).round(3).to_string(index=False, header=len(rows) == 1), flush=True)
    df = pd.DataFrame(rows)
    if out_path:
        df.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
