"""
Score mouse-brain fits against the single-cell long-read reference (truth.py): within-gene isoform fractions
per region x cell class, over genes with >= MIN_GENE_UMI truth UMIs on >= 2 reference transcripts and gene TPM
>= 1 in the estimate. Estimates: SR_only (mean of the region|class groups over ages), LR_bulk (the region's
bulk tissue profile, the same for every class), joint base unit and joint consensus. Metrics: median gene TV
of isoform fractions and Pearson of isoform fractions over all scored transcripts.

Run: python analyses/mouse_brain/evaluate.py <fit_dir> <truth.tsv.gz> <ref_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_GENE_UMI = 20


def within(est, truth, gene):
    """(median gene TV, psi Pearson, genes) between estimate and truth over eligible genes."""
    idx = truth.index.intersection(est.index)
    e, t, g = est.reindex(idx).fillna(0.0), truth.reindex(idx), gene.reindex(idx)
    gt, ge = t.groupby(g).transform("sum"), e.groupby(g).transform("sum")
    n = g.groupby(g).transform("size")
    ok = (gt >= MIN_GENE_UMI) & (ge >= 1.0) & (n >= 2)
    if not ok.any():
        return np.nan, np.nan, 0
    pe, pt = (e / ge)[ok], (t / gt)[ok]
    tv = (0.5 * (pe - pt).abs()).groupby(g[ok]).sum()
    return float(tv.median()), float(np.corrcoef(pe, pt)[0, 1]), int(tv.size)


def main(fit_dir, truth_path, ref_dir):
    fd = Path(fit_dir)
    tpm = pd.read_csv(fd / "tpm.tsv.gz", sep="\t", index_col=0)
    tpm = tpm[~tpm.index.str.endswith("-U")]
    truth = pd.read_csv(truth_path, sep="\t", index_col=0)
    t2g = pd.read_csv(Path(ref_dir) / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    gene = t2g.reindex(tpm.index)
    rows = []
    for col in [c for c in truth.columns if c.endswith("|all")]:
        region, cls = col.split("|")[:2]
        t = truth[col]
        cands = {}
        sr = [c for c in tpm.columns if c.startswith(f"SR_only|{region}|{cls}|")]
        if sr:
            cands["SR_only"] = tpm[sr].mean(axis=1)
        if f"LR_bulk|{region}" in tpm:
            cands["LR_bulk"] = tpm[f"LR_bulk|{region}"]
        for name, c in [("joint", f"joint|{region}|{cls}"), ("joint_consensus", f"joint|{region}|{cls}|consensus")]:
            if c in tpm:
                cands[name] = tpm[c]
        for name, est in cands.items():
            tv, r, ng = within(est, t, gene)
            rows.append({"region": region, "cell_class": cls, "estimate": name, "gene_tv_median": tv,
                         "psi_pearson": r, "genes": ng})
    r = pd.DataFrame(rows)
    r.to_csv(fd / "evaluation.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(r.pivot_table(index=["region", "cell_class"], columns="estimate", values="gene_tv_median").round(3).to_string())
    print(r.groupby("estimate")[["gene_tv_median", "psi_pearson", "genes"]].median().round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
