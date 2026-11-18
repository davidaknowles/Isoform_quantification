"""
Inputs for the Tabula Sapiens atlas pilot: Smart-seq2 pseudobulk groups and the matching ENCODE4 PacBio files.

Short reads: Tabula Sapiens v1 Smart-seq2 cells of the chosen tissues (one donor each: muscle TSP4, heart
TSP12), grouped by tissue x cell class; cell ontology classes are merged into classes (CLASS) and groups need
at least MIN_CELLS cells, with at most MAX_CELLS sampled (fixed seed). Each cell's read pair is located in the
S3 listing of its donor's Smart-seq2 fastqs (<cell_id>_R1/_R2.fastq.gz). Columns follow the mouse-brain
pilot so its library and fit scripts apply: region = tissue, cell_class, age = donor.

Long reads: ENCODE4 PacBio fastqs of the same tissues (psoas muscle; heart chambers and septum).

Writes <out_dir>/sr/groups.tsv, sr/cells.tsv (group, cell, r1 key, r2 key) and lr/files.tsv.

Run: python analyses/tabula_sapiens/make_groups.py <TSP1_TSP15_metadata.csv> <encode_dir> <out_dir> <s3_listing>...
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TISSUES = {"Muscle": "TSP4", "Heart": "TSP12"}
CLASS = {
    # muscle
    "mesenchymal stem cell": "mesenchymal stem cell", "skeletal muscle satellite stem cell": "satellite cell",
    "endothelial cell of vascular tree": "endothelial", "capillary endothelial cell": "endothelial",
    "endothelial cell of artery": "endothelial", "blood vessel endothelial cell": "endothelial",
    "endothelial cell of lymphatic vessel": "lymphatic endothelial", "cd8-positive, alpha-beta t cell": "T cell",
    "cd4-positive, alpha-beta t cell": "T cell", "t cell": "T cell", "mature nk t cell": "T cell",
    "macrophage": "macrophage", "pericyte cell": "pericyte", "fast muscle cell": "fast muscle cell",
    "slow muscle cell": "slow muscle cell", "mast cell": "mast cell", "smooth muscle cell": "smooth muscle cell",
    # heart
    "cardiac endothelial cell": "endothelial", "fibroblast of cardiac tissue": "fibroblast",
    "cardiac muscle cell": "cardiac muscle cell", "hepatocyte": "hepatocyte-labelled",
}
ENCODE_TISSUE = {"psoas muscle": "Muscle"}
HEART_TERMS = ("heart", "cardiac", "ventricle", "atrium", "myocardium")
MIN_CELLS, MAX_CELLS = 10, 300


def main(meta, encode_dir, out_dir, *listings):
    out = Path(out_dir)
    (out / "sr").mkdir(parents=True, exist_ok=True)
    (out / "lr").mkdir(parents=True, exist_ok=True)
    keys = pd.concat([pd.read_csv(f, sep=r"\s+", header=None, usecols=[3], names=["key"]) for f in listings])["key"]
    by_name = pd.Series(keys.to_numpy(), index=keys.str.rsplit("/", n=1).str[1])
    m = pd.read_csv(meta, low_memory=False)
    s = m[(m.method == "smartseq2") & m.tissue.isin(list(TISSUES)) & (m.donor == m.tissue.map(TISSUES))].copy()
    s["cell_class"] = s.cell_ontology_class.map(CLASS)
    s = s.dropna(subset=["cell_class"])
    s["r1"] = by_name.reindex(s.cell_id + "_R1.fastq.gz").to_numpy()
    s["r2"] = by_name.reindex(s.cell_id + "_R2.fastq.gz").to_numpy()
    print(f"{len(s)} cells, {s.r1.notna().sum()} with fastqs on S3")
    s = s.dropna(subset=["r1", "r2"])
    s["group"] = s.tissue + "|" + s.cell_class + "|" + s.donor
    rng = np.random.default_rng(0)
    keep = [d.iloc[rng.permutation(len(d))[:MAX_CELLS]] for _, d in s.groupby("group") if len(d) >= MIN_CELLS]
    cells = pd.concat(keep)
    groups = cells.groupby("group").agg(region=("tissue", "first"), cell_class=("cell_class", "first"),
                                        age=("donor", "first"), n_cells=("cell_id", "size")).reset_index()
    groups.to_csv(out / "sr" / "groups.tsv", sep="\t", index=False)
    cells[["group", "cell_id", "r1", "r2"]].to_csv(out / "sr" / "cells.tsv", sep="\t", index=False)
    print(groups.to_string(index=False))
    print(f"{len(groups)} groups, {len(cells)} cells")

    e = pd.read_csv(Path(encode_dir) / "metadata.tsv", sep="\t", low_memory=False)
    term = e["Biosample term name"]
    tissue = term.map(ENCODE_TISSUE).where(~term.str.contains("|".join(HEART_TERMS), case=False), "Heart")
    f = pd.DataFrame({"accession": e["File accession"], "tissue": tissue, "experiment": e["Experiment accession"],
                      "donor": e["Donor(s)"].str.strip("/").str.split("/").str[-1], "replicate": e["Biological replicate(s)"],
                      "platform": e["Platform"], "fastq": [str(Path(encode_dir) / f"{a}.fastq.gz") for a in e["File accession"]],
                      "term": term})
    f = f.dropna(subset=["tissue"])
    f = f[[Path(p).exists() for p in f.fastq]]
    f.to_csv(out / "lr" / "files.tsv", sep="\t", index=False)
    print(f.groupby("tissue").size().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4], *sys.argv[4:])
