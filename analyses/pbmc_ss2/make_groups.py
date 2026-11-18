"""
Smart-seq2 PBMC short-read groups (Ding et al. 2020, GSE132044; HCA per-cell HISAT2 BAMs) for comparison with
the 10x Kinnex single-cell long reads.

matrix: the Smart-seq2 cells of the GEO count matrix (cells passing the authors' QC) as a cell x gene matrix
        for analyses/kinnex_parse/celltypist_labels.py (CellTypist Immune_All_Low with majority voting, the
        labels of the Kinnex and Parse units), in <out_dir>/molecules.
groups: each BAM is linked to its cell through the HCA manifest (BAM name = sequencing process id; the same
        process's fastq names carry donor, plate and well, e.g. PBMC1.CC7W2ANXX.SM2_120717p3_C2 ->
        PBMC1.Smart-seq2.p3_C2); groups are cell type x donor with at least MIN_CELLS cells. Columns follow
        the mouse-brain pilot (region = PBMC, cell_class = cell type, age = donor).

Writes <out_dir>/sr/groups.tsv and sr/cells.tsv (group, cell, BAM).

Run: python analyses/pbmc_ss2/make_groups.py matrix <hca_dir> <out_dir>
     ~/venv/celltypist/bin/python analyses/kinnex_parse/celltypist_labels.py <out_dir>/molecules <out_dir>/celltypist.tsv 0
     python analyses/pbmc_ss2/make_groups.py groups <hca_dir> <out_dir>
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

GEO = "fae506c0-72ad-50b8-a2f4-91323a74dc2d"
MIN_CELLS = 10


def matrix(hca, out):
    g = Path(hca) / GEO
    X = sp.csc_matrix(scipy.io.mmread(g / "GSE132044_pbmc_hg38_count_matrix.mtx.gz"))      # genes x cells
    cells = pd.read_csv(g / "GSE132044_pbmc_hg38_cell.tsv.gz", header=None)[0]
    genes = pd.read_csv(g / "GSE132044_pbmc_hg38_gene.tsv.gz", header=None)[0].str.split("_", n=1).str[1]
    keep = np.flatnonzero(cells.str.contains(".Smart-seq2.", regex=False).to_numpy())
    m = Path(out) / "molecules"
    m.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(m / "counts.mtx", sp.csr_matrix(X[:, keep].T))
    (m / "cells.txt").write_text("\n".join(cells.iloc[keep]) + "\n")
    (m / "genes.txt").write_text("\n".join(genes) + "\n")
    print(f"{len(keep)} Smart-seq2 cells, {len(genes)} genes")


def groups(hca, out):
    hca, out = Path(hca), Path(out)
    man = pd.read_csv(hca / "manifest.tsv", sep="\t", low_memory=False)
    ss2 = man[(man["library_preparation_protocol.library_construction_approach"] == "Smart-seq2")
              & man["cell_suspension.biomaterial_core.biomaterial_id"].str.startswith("PBMC")]
    proc = "sequencing_process.provenance.document_id"
    fq = ss2[ss2.file_format == "fastq.gz"].drop_duplicates(proc)
    pat = fq.file_name.str.extract(r"^(PBMC\d)\.[^.]+\.SM2_\d+(p\d+)_([A-P]\d+)\.")
    cell_of = pd.Series((pat[0] + ".Smart-seq2." + pat[1] + "_" + pat[2]).to_numpy(), index=fq[proc].to_numpy())
    bam = ss2[ss2.file_format == "bam"]
    b = pd.DataFrame({"cell": cell_of.reindex(bam[proc]).to_numpy(),
                      "bam": [str(hca / u / f) for u, f in zip(bam.bundle_uuid, bam.file_name)]})
    b = b[[Path(p).exists() for p in b.bam]].dropna()
    ct = pd.read_csv(out / "celltypist.tsv", sep="\t").set_index("cell")["cell_type"]
    b["cell_type"] = ct.reindex(b.cell).to_numpy()
    print(f"{len(b)} BAMs linked to cells, {b.cell_type.notna().sum()} labelled (cells passing QC)")
    b = b.dropna(subset=["cell_type"])
    b["donor"] = b.cell.str.split(".").str[0]
    b["group"] = b.cell_type + "|" + b.donor
    n = b.groupby("group").size()
    b = b[b.group.isin(n.index[n >= MIN_CELLS])]
    (out / "sr").mkdir(parents=True, exist_ok=True)
    grp = b.groupby("group").agg(region=("donor", lambda x: "PBMC"), cell_class=("cell_type", "first"),
                                 age=("donor", "first"), n_cells=("bam", "size")).reset_index()
    grp.to_csv(out / "sr" / "groups.tsv", sep="\t", index=False)
    b[["group", "cell", "bam"]].to_csv(out / "sr" / "cells.tsv", sep="\t", index=False)
    print(grp.to_string(index=False))
    print(f"{len(grp)} groups, {len(b)} cells; all labels:")
    print(ct.value_counts().to_string())


if __name__ == "__main__":
    {"matrix": matrix, "groups": groups}[sys.argv[1]](*sys.argv[2:4])
