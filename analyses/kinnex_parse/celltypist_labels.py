"""
CellTypist labels (Immune_All_Low with majority voting, the model tealeaf used
for the Parse PBMC short-read series) for cells from a cell x gene UMI matrix.

Needs the CellTypist venv (Python 3.12 module loaded).

Run: python analyses/kinnex_parse/celltypist_labels.py <molecules_dir> <out.tsv> [min_umis]
"""

import sys
from pathlib import Path

import anndata
import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io


def main(mdir, out, min_umis=200):
    mdir = Path(mdir)
    X = scipy.io.mmread(mdir / "counts.mtx").tocsr().astype(np.float32)
    cells = (mdir / "cells.txt").read_text().split()
    genes = (mdir / "genes.txt").read_text().split()
    ad = anndata.AnnData(X=X, obs=pd.DataFrame(index=cells[: X.shape[0]]), var=pd.DataFrame(index=genes))
    ad.var_names_make_unique()
    ad.obs["umis"] = np.asarray(X.sum(1)).ravel()
    ad = ad[ad.obs["umis"] >= float(min_umis)].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    pred = celltypist.annotate(ad, model="Immune_All_Low.pkl", majority_voting=True)
    lab = pred.predicted_labels
    res = pd.DataFrame({"cell": ad.obs_names, "umis": ad.obs["umis"].to_numpy(),
                        "cell_type": lab["majority_voting"].astype(str).str.replace(" ", "_").to_numpy(),
                        "predicted": lab["predicted_labels"].astype(str).to_numpy()})
    res.to_csv(out, sep="\t", index=False)
    print(res["cell_type"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
