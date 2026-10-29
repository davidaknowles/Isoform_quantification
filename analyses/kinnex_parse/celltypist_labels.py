"""
CellTypist labels (Immune_All_Low with majority voting, the model tealeaf used
for the Parse PBMC short-read series) for cells from a cell x gene UMI matrix.
Majority voting runs over a PCA + k-means over-clustering (~25 cells per cluster).

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
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans


def main(mdir, out, min_umis=200):
    mdir = Path(mdir)
    # scanpy here needs the spmatrix API (newer scipy returns csr_array from mmread)
    X = sp.csr_matrix(scipy.io.mmread(mdir / "counts.mtx")).astype(np.float32)
    cells = (mdir / "cells.txt").read_text().split()
    genes = (mdir / "genes.txt").read_text().split()
    ad = anndata.AnnData(X=X, obs=pd.DataFrame(index=cells[: X.shape[0]]), var=pd.DataFrame(index=genes))
    ad.var_names_make_unique()
    ad.obs["umis"] = np.asarray(X.sum(1)).ravel()
    ad = ad[ad.obs["umis"] >= float(min_umis)].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    # majority voting over our own over-clustering (PCA + k-means, ~25 cells per cluster):
    # CellTypist's default builds a neighbour graph with a scanpy that fails on this scipy
    pca = sc.pp.pca(ad, n_comps=50, copy=True).obsm["X_pca"]
    k = max(2, ad.n_obs // 25)
    clusters = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit_predict(pca)
    pred = celltypist.annotate(ad, model="Immune_All_Low.pkl", majority_voting=True,
                               over_clustering=clusters.astype(str))
    lab = pred.predicted_labels
    res = pd.DataFrame({"cell": ad.obs_names, "umis": ad.obs["umis"].to_numpy(),
                        "cell_type": lab["majority_voting"].astype(str).str.replace(" ", "_").to_numpy(),
                        "predicted": lab["predicted_labels"].astype(str).to_numpy()})
    res.to_csv(out, sep="\t", index=False)
    print(res["cell_type"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
