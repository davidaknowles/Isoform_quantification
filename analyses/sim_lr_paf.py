"""
Long-read input from minimap2 transcriptome alignments (best-score ties, with
and without the read-end truncation model; and the 0.95 score threshold)
versus lr-kallisto TCCs on the simulation, in
LR-only and SR+LR (salmon SR, no bias correction) fits.

minimap2 was run as
    minimap2 -c -x map-hifi -N 100 transcriptome.fasta reads.fasta > reads.paf

Usage: python analyses/sim_lr_paf.py <paf_dir> <salmon_quant_root> [out.tsv] [samples]
    paf_dir contains {sim1,sim2}_long.paf.gz; samples defaults to sim1,sim2
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.evaluate import fraction_metrics, read_t2g, read_tpm, transcript_metrics
from joli.joint import JointEM, library_from_tcc, molar_to_tpm
from joli.load_tcc import load_tcc_data
from joli.lr_bias import TruncationModel, fit_with_truncation
from joli.paf import filter_alignments, library_from_alignments, read_paf
from joli.salmon_io import library_from_salmon
from joli.weights import load_fasta_transcript_lengths

sys.path.insert(0, str(Path(__file__).parent))
from sim_joint import LK, SAMPLES, SIM, quiet  # noqa: E402


def main(paf_dir, salmon_root, out_path=None, samples=None):
    t2g = read_t2g(LK / "t2g.txt")
    rows = []
    samples = samples or list(SAMPLES)

    def emit(row, tpm, truth):
        row.update(transcript_metrics(tpm, truth))
        row.update(fraction_metrics(tpm, truth, t2g))
        rows.append(row)
        print(pd.DataFrame([row]).round(3).to_string(index=False, header=len(rows) == 1), flush=True)

    for sample in samples:
        stem, k = SAMPLES[sample]
        truth = read_tpm(SIM / "ground_truths" / f"sample{k}_gt.tsv")
        lr_tcc = quiet(load_tcc_data, SIM / "kallisto_output" / f"{stem}_long")
        names = lr_tcc.transcript_names
        tx_len = quiet(load_fasta_transcript_lengths, LK / "transcriptome.fasta", names)
        sr = library_from_salmon("sr", Path(salmon_root) / f"{sample}_nobias", transcript_names=names)

        t0 = time.time()
        raw = read_paf(Path(paf_dir) / f"{sample}_long.paf.gz")
        aln = filter_alignments(raw)
        print(f"{sample}: {aln['read'].nunique()} reads, {len(aln)} alignments ({time.time() - t0:.0f} s)", flush=True)
        model = TruncationModel(tx_len)
        feat = model.features(aln["tstart"], aln["tend"], aln["tlen"])
        lrs = {
            "lr-kallisto TCC": library_from_tcc("lr", lr_tcc, "long"),
            "minimap2 (AS >= 0.95 best)": library_from_alignments("lr", filter_alignments(raw, score_frac=0.95), names),
            "minimap2": library_from_alignments("lr", aln, names),
            "minimap2+trunc": library_from_alignments("lr", aln, names, features=feat),
        }
        for lr_name, lr in lrs.items():
            for fit, libs in {"LR": [lr], "SR+LR": [sr, lr]}.items():
                t0 = time.time()
                em = JointEM(libs, units=[0] * len(libs))
                if lr_name.endswith("+trunc"):
                    m = TruncationModel(tx_len)
                    res = fit_with_truncation(em, {len(libs) - 1: m}, n_outer=8, verbose=True)
                    print("  p5:", np.round(m.end5.p, 3), "\n  p3:", np.round(m.end3.p, 3), flush=True)
                else:
                    res = em.run()
                tpm = pd.Series(molar_to_tpm(res.theta[0]), index=names)
                emit({"sample": sample, "lr_input": lr_name, "fit": fit, "n_iter": res.n_iter,
                      "seconds": round(time.time() - t0, 1)}, tpm, truth)
    df = pd.DataFrame(rows)
    if out_path:
        df.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
         sys.argv[4].split(",") if len(sys.argv) > 4 else None)
