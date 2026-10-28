"""
Score SG-NEx fits on spike-ins with known relative molarity
(spikein_truth.py): H9 carries SIRV Set 4, HEYA8 carries sequins Mix A v2
and SIRV Set 1 mix E2.

Per fit, replicate and mix: within-gene isoform fractions against truth
(gene TV median over genes with >= 2 isoforms), and abundance against truth
across the mix's transcripts (Pearson of log values, SD of log ratio after
removing the mean; estimates floored at 1e-3 of the median nonzero one).

Run: python analyses/sgnex/spikein_eval.py <quant_dir> <spikein_truth.tsv> [out.tsv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIXES = {"H9": ["sirv4"], "HEYA8": ["sequinA_v2", "sirv_E2"]}


def score(est, truth):
    """est: Series over the mix's transcripts; truth: DataFrame (transcript index; gene, rel_molarity)."""
    x = est.reindex(truth.index).fillna(0.0)
    m = truth["rel_molarity"]
    gene = truth["gene"]
    n = gene.groupby(gene).transform("size")
    multi = n >= 2
    psi_e = (x / x.groupby(gene).transform("sum")).fillna(1.0 / n)
    psi_t = m / m.groupby(gene).transform("sum")
    tv = (psi_e - psi_t)[multi].abs().groupby(gene[multi]).sum() / 2.0
    floor = 1e-3 * x[x > 0].median() if (x > 0).any() else 1e-12
    lr = np.log(x + floor) - np.log(m)
    return {"gene_tv_median": float(tv.median()), "gene_tv_mean": float(tv.mean()),
            "log_pearson": float(np.corrcoef(np.log(x + floor), np.log(m))[0, 1]),
            "sd_log_ratio": float(lr.std()), "n_zero": int((x == 0).sum())}


def main(quant_dir, truth_path, out_path=None):
    truth = pd.read_csv(truth_path, sep="\t")
    rows = []
    for f in sorted(Path(quant_dir).glob("*_tpm.tsv.gz")):
        cell, rep = f.name.split("_")[:2]
        df = pd.read_csv(f, sep="\t", index_col=0)
        for mix in MIXES[cell]:
            t = truth[truth["mix"] == mix].set_index("transcript")
            for fit in df.columns:
                rows.append({"cell": cell, "rep": rep, "mix": mix, "fit": fit, **score(df[fit], t)})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.groupby(["mix", "fit"], sort=False)[["gene_tv_median", "log_pearson", "sd_log_ratio", "n_zero"]]
          .mean().round(3).to_string())
    if out_path:
        out.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
