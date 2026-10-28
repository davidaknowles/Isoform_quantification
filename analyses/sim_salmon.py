"""
Short-read input from salmon (selective alignment, rich EC weights, optional
sequence/GC/positional bias correction) versus kallisto pseudoalignment, in
SR-only and SR+LR JointEM fits on the LRGASP-style simulation. salmon's own
quant.sf estimate is scored as a baseline.

salmon was run as
    salmon quant -l A --validateMappings --dumpEq --dumpEqWeights --useEM [--seqBias --gcBias --posBias]
against an index of the same transcript set as the kallisto index.

Usage: python analyses/sim_salmon.py <salmon_quant_root> [out.tsv]
    salmon_quant_root contains {sim1,sim2}_{nobias,bias}/
"""

import sys
import time
from pathlib import Path

import pandas as pd

from joli.evaluate import fraction_metrics, read_t2g, read_tpm, transcript_metrics
from joli.joint import JointEM, library_from_tcc, molar_to_tpm
from joli.load_tcc import load_tcc_data
from joli.salmon_io import library_from_salmon, read_salmon_quant
from joli.weights import (calc_eff_lens_from_fld, compute_mean_fl_trunc,
                          load_fasta_transcript_lengths, load_fld_histogram)

sys.path.insert(0, str(Path(__file__).parent))
from sim_joint import LK, SAMPLES, SIM, quiet  # noqa: E402


def score(row, tpm, truth, t2g):
    row.update(transcript_metrics(tpm, truth))
    row.update(fraction_metrics(tpm, truth, t2g))
    return row


def main(salmon_root, out_path=None):
    salmon_root = Path(salmon_root)
    t2g = read_t2g(LK / "t2g.txt")
    rows = []

    def emit(row):
        rows.append(row)
        print(pd.DataFrame([row]).round(3).to_string(index=False, header=len(rows) == 1), flush=True)

    tx_len = None
    for sample, (stem, k) in SAMPLES.items():
        truth = read_tpm(SIM / "ground_truths" / f"sample{k}_gt.tsv")
        lr_tcc = quiet(load_tcc_data, SIM / "kallisto_output" / f"{stem}_long")
        names = lr_tcc.transcript_names
        lr = library_from_tcc("lr", lr_tcc, "long")
        sr_dir = SIM / "kallisto_output" / f"{stem}_short_1"
        sr_tcc = quiet(load_tcc_data, sr_dir)
        if tx_len is None:
            tx_len = quiet(load_fasta_transcript_lengths, LK / "transcriptome.fasta", names)
        eff = quiet(calc_eff_lens_from_fld, tx_len, quiet(compute_mean_fl_trunc, quiet(load_fld_histogram, sr_dir / "flens.txt")))
        srs = {"kallisto": library_from_tcc("sr", sr_tcc, "short", eff_lens=eff)}
        for mode in ["nobias", "bias"]:
            qdir = salmon_root / f"{sample}_{mode}"
            q = read_salmon_quant(qdir)
            emit(score({"sample": sample, "sr_input": f"salmon_{mode}", "fit": "salmon quant.sf"},
                       q["TPM"].reindex(names).fillna(0.0), truth, t2g))
            srs[f"salmon_{mode}"] = library_from_salmon("sr", qdir, transcript_names=names)
            srs[f"salmon_{mode}_noweights"] = library_from_salmon("sr", qdir, transcript_names=names,
                                                                  use_weights=False)
        for sr_name, sr in srs.items():
            for fit, libs in {"SR": [sr], "SR+LR": [sr, lr]}.items():
                t0 = time.time()
                res = JointEM(libs, units=[0] * len(libs)).run(max_iter=5000)
                tpm = pd.Series(molar_to_tpm(res.theta[0]), index=names)
                emit(score({"sample": sample, "sr_input": sr_name, "fit": fit, "n_iter": res.n_iter,
                            "seconds": round(time.time() - t0, 1)}, tpm, truth, t2g))
    df = pd.DataFrame(rows)
    if out_path:
        df.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
