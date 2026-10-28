"""
Cross-platform and cross-replicate agreement of WTC11 fits from quant.py.

With no ground truth, a bias model that removes platform-specific
distortion should bring estimates from different platforms of the same
biosample closer together, and should not reduce agreement between
biosamples on one platform.

Run: python analyses/wtc11/compare.py <quant_dir> <t2g.tsv> [out_prefix]
"""

import itertools
import sys
from pathlib import Path

import pandas as pd

from joli.evaluate import concordance_metrics, read_t2g

BIOSAMPLES = ["ENCBS944CBA", "ENCBS593PKA", "ENCBS474NOC"]
PLATFORM_PAIRS = [
    ("SR:nobias", "PB"), ("SR:bias", "PB"), ("SR:nobias", "PB+trunc"), ("SR:bias", "PB+trunc"),
    ("SR:nobias", "CT"), ("SR:bias", "CT+trunc"),
    ("SR:nobias", "ONT"), ("SR:bias", "ONT+trunc"),
    ("PB", "CT"), ("PB+trunc", "CT+trunc"),
    ("PB", "ONT"), ("PB+trunc", "ONT+trunc"),
    ("CT", "ONT"), ("CT+trunc", "ONT+trunc"),
    ("SR:bias+PB+trunc", "ONT+trunc"), ("SR:bias+PB+trunc", "CT+trunc"),
    ("SR:bias+CT+trunc", "PB+trunc"), ("SR:bias+ONT+trunc", "PB+trunc"),
]


def main(quant_dir, t2g_path, out_prefix=None):
    quant_dir = Path(quant_dir)
    t2g = read_t2g(t2g_path)
    tpm = {bs: pd.read_csv(quant_dir / f"{bs}_tpm.tsv.gz", sep="\t", index_col=0)
           for bs in BIOSAMPLES if (quant_dir / f"{bs}_tpm.tsv.gz").exists()}
    rows = []
    for bs, df in tpm.items():
        for a, b in PLATFORM_PAIRS:
            if a in df and b in df:
                rows.append({"comparison": "platform", "biosample": bs, "a": a, "b": b,
                             **concordance_metrics(df[a], df[b], t2g)})
    platform = pd.DataFrame(rows)
    rows = []
    for (b1, d1), (b2, d2) in itertools.combinations(tpm.items(), 2):
        for fit in d1.columns:
            rows.append({"comparison": "replicate", "fit": fit, "a": b1, "b": b2,
                         **concordance_metrics(d1[fit], d2[fit], t2g)})
    replicate = pd.DataFrame(rows)
    cols = ["pearson_log1p", "psi_pearson", "gene_tv_median", "major_isoform_agree"]
    if len(platform):
        print(platform.groupby(["a", "b"], sort=False)[cols].mean().round(3).to_string())
    if len(replicate):
        print(replicate.groupby("fit", sort=False)[cols].mean().round(3).to_string())
    if out_prefix:
        platform.to_csv(f"{out_prefix}_platform.tsv", sep="\t", index=False)
        replicate.to_csv(f"{out_prefix}_replicate.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
