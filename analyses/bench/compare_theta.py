"""
Compare design-fit solutions saved by design_convergence.py against a reference (theta_<name>.npz):
per unit, TPM total variation, log-Pearson over transcripts > 1 TPM, and median within-gene TV of
isoform fractions (spliced genes > 10 TPM); summarised over units.

Run: python analyses/bench/compare_theta.py <dir> <reference> <other> [<other> ...]
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(d, ref, *others):
    d = Path(d)
    with open(d / "prepared.pkl", "rb") as fh:
        ctx = pickle.load(fh)["ctx"]
    genes, uns = np.asarray(ctx["genes"]), np.asarray(ctx["unspliced"])
    z = np.load(d / f"theta_{ref}.npz")
    R, units = z["theta"], list(z["present"])
    rows = []
    for o in others:
        zo = np.load(d / f"theta_{o}.npz")
        O = zo["theta"][[list(zo["present"]).index(u) for u in units]]
        for i, u in enumerate(units):
            a, b = 1e6 * R[i] / R[i].sum(), 1e6 * O[i] / O[i].sum()
            ok = (a > 1) | (b > 1)
            sp = ~uns & (genes >= 0)
            ga = pd.Series(a[sp]).groupby(genes[sp]).transform("sum").to_numpy()
            gb = pd.Series(b[sp]).groupby(genes[sp]).transform("sum").to_numpy()
            keep = (ga > 10) & (gb > 10)
            tv = pd.Series(0.5 * np.abs(a[sp][keep] / ga[keep] - b[sp][keep] / gb[keep])).groupby(genes[sp][keep]).sum()
            rows.append({"variant": o, "unit": u, "tpm_tv": 0.5 * np.abs(a - b).sum() / 1e6,
                         "log_pearson": np.corrcoef(np.log1p(a[ok]), np.log1p(b[ok]))[0, 1],
                         "gene_tv_median": tv.median(), "gene_tv_p90": tv.quantile(0.9)})
    r = pd.DataFrame(rows)
    print(r.groupby("variant")[["tpm_tv", "log_pearson", "gene_tv_median", "gene_tv_p90"]].agg(["mean", "max"]).round(4).to_string())
    r.to_csv(d / f"compare_vs_{ref}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
