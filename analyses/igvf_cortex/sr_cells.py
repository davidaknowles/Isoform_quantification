"""
Cells of the IGVF B01_13H cortex sublibrary from the short-read alevin-fry quantification (sr_align.sh).

Each short-read barcode (bc3 + bc2 + bc1 as read 2 reads them) is mapped to its round-1 well, RT primer and mouse
(subpool barcode map) and to the round-2/3 indices in the v1 onlist; the cell is <bc1 well>_<bc2 index>_<bc3 index>,
the name ont_barcodes.py gives long reads, so a well's oligo-dT and random-hexamer barcodes form one cell. Cells of
the B6J mice (GRCm39 reference) with at least MIN_UMIS short-read UMIs are kept; their gene counts (transcripts summed
to gene symbols) go to <igvf_dir>/molecules_sr for CellTypist.

Writes <igvf_dir>/sr_barcodes.tsv (SR barcode, cell, primer, mouse, umis), <igvf_dir>/cells.tsv (barcode = cell, cell,
mouse, sr_umis; the long-read cell table read by kinnex_parse/molecules.py) and molecules_sr/{counts.mtx,cells.txt,genes.txt}.

Run: python analyses/igvf_cortex/sr_cells.py <igvf_dir> <ref_dir>
     ~/venv/celltypist/bin/python analyses/kinnex_parse/celltypist_labels.py <igvf_dir>/molecules_sr \
         <igvf_dir>/celltypist.tsv 0 Mouse_Isocortex_Hippocampus.pkl
"""

import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

MIN_UMIS = 500
STRAIN = "B6J"


def gene_names(ref_dir):
    """transcript -> gene symbol from basic.gtf (unspliced targets <gene_id>-U map to the gene's symbol)."""
    tx_name = {}
    gid_name = {}
    with open(Path(ref_dir) / "basic.gtf") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if f[2] != "transcript":
                continue
            gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
            name = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
            tx_name[re.search(r'transcript_id "([^"]+)"', f[8]).group(1)] = name
            gid_name[gid] = name
    tx_name.update({f"{g}-U": n for g, n in gid_name.items()})
    return tx_name


def main(igvf_dir, ref_dir):
    d = Path(igvf_dir)
    q = d / "sr_quant" / "quant" / "alevin"
    X = sp.csr_matrix(scipy.io.mmread(q / "quants_mat.mtx"))
    bcs = (q / "quants_mat_rows.txt").read_text().split()
    txs = (q / "quants_mat_cols.txt").read_text().split()
    v1 = [w.strip() for w in gzip.open(d / "meta" / "IGVFFI1138MCVX.txt.gz", "rt") if w.strip()]
    idx = {w: i for i, w in enumerate(v1)}
    bmap = pd.read_csv(d / "meta" / "IGVFFI9171FPFR.tsv.gz", sep="\t").set_index("barcode")
    rows = []
    for j, b in enumerate(bcs):
        b3, b2, b1 = b[:8], b[8:16], b[16:24]
        if b1 not in bmap.index or b2 not in idx or b3 not in idx:
            continue
        m = bmap.loc[b1]
        rows.append({"row": j, "barcode": b, "cell": f"{m['well']}_{idx[b2]}_{idx[b3]}",
                     "primer": {"T": "poly(dT)", "R": "random hexamer"}[m["parse barcode type"]],
                     "mouse": m["sample description"]})
    bc = pd.DataFrame(rows)
    bc["umis"] = np.asarray(X[bc["row"].to_numpy()].sum(axis=1)).ravel()
    print(f"{len(bcs)} called barcodes, {len(bc)} on the whitelists", flush=True)
    per_cell = bc.groupby("cell").agg(mouse=("mouse", "first"), sr_umis=("umis", "sum")).reset_index()
    keep = per_cell[(per_cell["sr_umis"] >= MIN_UMIS) & per_cell["mouse"].str.contains(STRAIN)]
    print(per_cell.groupby(per_cell["mouse"]).agg(cells=("cell", "size"),
                                                  kept=("sr_umis", lambda u: int((u >= MIN_UMIS).sum()))).to_string(), flush=True)
    bc = bc[bc["cell"].isin(keep["cell"])]
    bc.drop(columns="row").to_csv(d / "sr_barcodes.tsv", sep="\t", index=False)
    keep.assign(barcode=keep["cell"])[["barcode", "cell", "mouse", "sr_umis"]].to_csv(d / "cells.tsv", sep="\t", index=False)

    # cell x gene-symbol counts (oligo-dT and random-hexamer barcodes summed)
    cells = keep["cell"].tolist()
    ci = pd.Series(np.arange(len(cells)), index=cells)
    agg = sp.csr_matrix((np.ones(len(bc)), (ci[bc["cell"]].to_numpy(), bc["row"].to_numpy())), shape=(len(cells), X.shape[0]))
    C = agg @ X
    tx_name = gene_names(ref_dir)
    gsym = pd.Series([tx_name.get(t, "") for t in txs])
    codes, genes = pd.factorize(gsym)
    G = sp.csr_matrix((np.ones(len(txs)), (np.arange(len(txs)), codes)), shape=(len(txs), len(genes)))
    M = sp.csr_matrix(C @ G)
    ok = np.asarray(genes != "")
    out = d / "molecules_sr"
    out.mkdir(exist_ok=True)
    scipy.io.mmwrite(out / "counts.mtx", M[:, ok])
    (out / "cells.txt").write_text("\n".join(cells) + "\n")
    (out / "genes.txt").write_text("\n".join(np.asarray(genes)[ok]) + "\n")
    print(f"{len(cells)} {STRAIN} cells with >= {MIN_UMIS} UMIs; median UMIs {int(keep['sr_umis'].median())}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
