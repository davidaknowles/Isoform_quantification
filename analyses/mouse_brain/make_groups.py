"""
Short-read pseudobulk groups for the mouse-brain pilot from Tabula Muris Senis FACS Smart-seq2 cells.

Brain cells (Brain_Myeloid and Brain_Non-Myeloid sorts) with a BAM on disk are grouped by region
(subtissue: Cortex, Hippocampus, Striatum, Cerebellum), coarse cell class and age. Classes match the
cell types of the single-cell long-read reference (Joglekar et al. 2024): microglia (microglial cell,
macrophage), endothelial, pericyte, oligodendrocyte, OPC, astrocyte (astrocyte, Bergmann glial cell),
neuron (neuron, interneuron, medium spiny neuron); other labels are left out. Groups need at least
MIN_CELLS cells; at most MAX_CELLS are sampled (fixed seed).

Writes <out_dir>/groups.tsv (one row per group) and cells.tsv (group, cell, BAM).

Run: python analyses/mouse_brain/make_groups.py <tms_bams.tsv.gz> <out_dir>
    tms_bams.tsv.gz: one row per BAM with clean_cell_id, bam, tissue, subtissue, age, cell_ontology_class
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASS = {"microglial cell": "microglia", "macrophage": "microglia", "endothelial cell": "endothelial",
         "brain pericyte": "pericyte", "oligodendrocyte": "oligodendrocyte",
         "oligodendrocyte precursor cell": "OPC", "astrocyte": "astrocyte", "Bergmann glial cell": "astrocyte",
         "neuron": "neuron", "interneuron": "neuron", "medium spiny neuron": "neuron"}
MIN_CELLS, MAX_CELLS = 10, 300


def main(bams, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j = pd.read_csv(bams, sep="\t")
    b = j[j.tissue.isin(["Brain_Myeloid", "Brain_Non-Myeloid"])].copy()
    b["region"] = b.subtissue.str.strip()
    b["cell_class"] = b.cell_ontology_class.map(CLASS)
    b = b.dropna(subset=["cell_class", "region"])
    b["group"] = b.region + "|" + b.cell_class + "|" + b.age
    rng = np.random.default_rng(0)
    keep = []
    for g, d in b.groupby("group"):
        if len(d) < MIN_CELLS:
            continue
        keep.append(d.iloc[rng.permutation(len(d))[:MAX_CELLS]])
    cells = pd.concat(keep)
    groups = cells.groupby("group").agg(region=("region", "first"), cell_class=("cell_class", "first"),
                                        age=("age", "first"), n_cells=("bam", "size")).reset_index()
    groups.to_csv(out / "groups.tsv", sep="\t", index=False)
    cells[["group", "clean_cell_id", "bam"]].to_csv(out / "cells.tsv", sep="\t", index=False)
    print(groups.to_string(index=False))
    print(f"{len(groups)} groups, {len(cells)} cells")


if __name__ == "__main__":
    main(*sys.argv[1:3])
