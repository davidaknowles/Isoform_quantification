"""
Cell-class composition of the bulk long-read tissues by gene-level deconvolution (joli.deconv.mixture_weights).

Bulk gene profile: the LR_bulk fit of each tissue (molar fractions, spliced targets, summed per gene).
Reference profiles: the gene-level SR profiles of the tissue's region x class groups (fit.py sr_genes.tsv.gz,
from single-gene ECs; mean over ages) and, for cortex, Allen MOp SMART-seq glutamatergic and GABAergic neurons (exon read counts
summed per class and divided by the median spliced transcript length of the gene, so on a molecule scale;
the Tabula Muris Senis cortex neurons express little Slc17a7 and do not represent cortical excitatory
neurons). Neuron components are summed into the neuron class. Weights are fractions of bulk molecules.

Writes <out.tsv>: tissue, unit (region|class), weight; and component-level weights to <out>.components.tsv.

Run: python analyses/mouse_brain/composition.py <fit_dir> <ref_dir> <allen_cells_dir | -> <out.tsv>
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from joli.deconv import mixture_weights

MOLECULES = 1e6          # scale of the bulk pseudo-counts (does not change the weights)


def gtf_genes(gtf):
    """gene_id -> gene_name, and gene_id -> median spliced transcript length, from a GTF."""
    names, tx_len, tx_gene = {}, {}, {}
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if f[2] not in ("gene", "exon"):
                continue
            gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
            if f[2] == "gene":
                names[gid] = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
            else:
                tid = re.search(r'transcript_id "([^"]+)"', f[8]).group(1)
                tx_len[tid] = tx_len.get(tid, 0) + int(f[4]) - int(f[3]) + 1
                tx_gene[tid] = gid
    glen = pd.Series(tx_len).groupby(pd.Series(tx_gene)).median()
    return pd.Series(names), glen


def allen_profiles(d, names, glen):
    """Allen MOp glutamatergic and GABAergic neuron gene profiles on a molecule scale (symbol index)."""
    ann = pd.read_csv(d / "cluster.annotation.csv").set_index("cluster_id")
    mem = pd.read_csv(d / "cluster.membership.csv", index_col=0)["x"]
    cls = ann["class_label"].reindex(mem.to_numpy()).set_axis(mem.index)
    counts = pd.read_csv(d / "exon.counts.csv.gz", index_col=0)
    allen = {}
    sym_len = glen.groupby(names.reindex(glen.index).to_numpy()).median()
    for c, label in [("Glutamatergic", "neuron:Allen_glutamatergic"), ("GABAergic", "neuron:Allen_GABAergic")]:
        cells = cls.index[cls == c].intersection(counts.columns)
        allen[label] = counts[cells].sum(axis=1) / sym_len.reindex(counts.index)
    return pd.DataFrame(allen)


def main(fit_dir, ref_dir, allen_dir, out):
    names, glen = gtf_genes(Path(ref_dir) / "basic.gtf")
    tpm = pd.read_csv(Path(fit_dir) / "tpm.tsv.gz", sep="\t", index_col=0,
                      usecols=lambda c: c == "Unnamed: 0" or c.startswith("LR_bulk|"))
    tpm = tpm[~tpm.index.str.endswith("-U")]
    t2g = pd.read_csv(Path(ref_dir) / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    sym = names.reindex(t2g.reindex(tpm.index)).to_numpy()
    G = tpm[pd.notna(sym)].groupby(sym[pd.notna(sym)]).sum()               # gene symbol x LR_bulk tissue
    srg = pd.read_csv(Path(fit_dir) / "sr_genes.tsv.gz", sep="\t", index_col=0)
    ssym = names.reindex(t2g.reindex(srg.index)).to_numpy()
    S = srg[pd.notna(ssym)].groupby(ssym[pd.notna(ssym)]).sum()              # gene symbol x SR group

    allen = pd.DataFrame()
    if allen_dir != "-":
        allen = allen_profiles(Path(allen_dir), names, glen)

    rows, comp_rows = [], []
    for col in [c for c in G.columns if c.startswith("LR_bulk|")]:
        tissue = col.split("|")[1]
        ref = {}
        for c in sorted({c.split("|")[1] for c in S.columns if c.startswith(f"{tissue}|")}):
            ref[f"{c}:SR"] = S[[x for x in S.columns if x.startswith(f"{tissue}|{c}|")]].mean(axis=1)
        ref = pd.DataFrame(ref)
        if tissue == "Cortex" and len(allen):
            ref = ref.join(allen, how="inner")
        ref = ref.dropna()
        ref = ref[(ref.sum(axis=1) > 0)]
        y = G[col].reindex(ref.index).fillna(0.0)
        w, ll = mixture_weights(y.to_numpy() / y.sum() * MOLECULES, ref.T.to_numpy())
        comp = pd.Series(w, index=ref.columns)
        for k, v in comp.items():
            comp_rows.append({"tissue": tissue, "component": k, "weight": v})
        unit = comp.groupby(comp.index.str.split(":").str[0]).sum()
        for c, v in unit.items():
            rows.append({"tissue": tissue, "unit": f"{tissue}|{c}", "weight": v})
        print(tissue, f"{len(ref)} genes", comp.round(3).to_dict(), flush=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    pd.DataFrame(comp_rows).to_csv(str(out).replace(".tsv", "") + ".components.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:5])
