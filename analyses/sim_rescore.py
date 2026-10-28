"""
Re-score the inherited long-read simulation runs (lr-kallisto, JK single-sample,
JK multi-sample) against both the generating abundances (sample*_gt.tsv) and
the realized long-read counts (PB_sample*_gt.tsv) used previously.

Usage: python analyses/sim_rescore.py [out.tsv]
"""

import sys
from pathlib import Path

import pandas as pd

from joli.evaluate import fraction_metrics, read_t2g, read_tpm, transcript_metrics

RESULTS = Path("/gpfs/commons/home/atalukder/RNA_Splicing/files/results")
GT_DIR = Path("/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/sim_real_data/ground_truths")
T2G = "/gpfs/commons/home/atalukder/RNA_Splicing/data/Shree_stuff/SOTA/lr-kallisto/t2g.txt"

RUNS = {
    "lr-kallisto": "exprmnt_2026_05_18__12_42_22",
    "JK_SS": "exprmnt_2026_05_18__12_46_25",
    "JK_MS": "exprmnt_2026_05_20__13_04_38",
}
SAMPLES = {"sim1": "1", "sim2": "2"}
TRUTHS = {"generating": "sample{}_gt.tsv", "PB_reads": "PB_sample{}_gt.tsv"}


def main(out_path=None):
    t2g = read_t2g(T2G)
    rows = []
    for method, exp in RUNS.items():
        for sample, k in SAMPLES.items():
            est = read_tpm(RESULTS / exp / sample / "abundance.tsv")
            for truth_name, pattern in TRUTHS.items():
                truth = read_tpm(GT_DIR / pattern.format(k))
                row = {"method": method, "sample": sample, "truth": truth_name}
                row.update(transcript_metrics(est, truth))
                row.update(fraction_metrics(est, truth, t2g))
                rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    if out_path:
        df.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
