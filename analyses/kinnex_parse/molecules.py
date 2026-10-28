"""
Per-molecule alignments and cell x gene counts for the Kinnex Parse long reads.

Stages:
  chunk <k>  cdna_<k> alignments (minimap2 PAF) filtered to best-score ties on
             the sense strand, joined with the read's cell, UMI and primer
             (called cells only); saved as molecules/chunk_<k>.npz.
  combine    one read per molecule (cell, UMI, primer), the one with the
             longest cDNA; saves molecules/alignments.npz (per alignment: mol,
             cell, half, tx_idx, tstart, tend, tlen) and a cell x gene UMI
             matrix (counts.mtx, cells.txt, genes.txt; gene symbols) for
             CellTypist, counting molecules whose alignments share one gene.

Transcript indices follow the alignment reference (spliceu.fa order).

Run: python analyses/kinnex_parse/molecules.py <kinnex_dir> <ref_dir> chunk <k>
     python analyses/kinnex_parse/molecules.py <kinnex_dir> <ref_dir> combine
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from joli.paf import filter_alignments, read_fasta_lengths, read_paf
from joli.rad import encode_2bit


def chunk(kdir, ref_dir, k):
    names, _ = read_fasta_lengths(ref_dir / "spliceu.fa")
    pos = pd.Series(np.arange(len(names), dtype=np.int32), index=names)
    cells = pd.read_csv(kdir / "cells.tsv", sep="\t")
    cell_ids = pd.Series(np.arange(cells["cell"].nunique()), index=pd.unique(cells["cell"]))
    bc_cell = pd.Series(cell_ids[cells["cell"]].to_numpy(), index=cells["barcode"])
    reads = pd.read_csv(kdir / "reads" / f"reads_{k}.tsv.gz", sep="\t",
                        usecols=["read", "cb", "umi", "primer", "cdna_len"])
    reads["cell"] = bc_cell.reindex(reads["cb"]).to_numpy()
    reads = reads.dropna(subset=["cell"]).set_index("read")
    aln = filter_alignments(read_paf(kdir / "minimap2" / f"cdna_{k}.paf.gz"), strand="+")
    rid = aln["read"].astype(str).astype(np.int64).to_numpy()
    ok = np.isin(rid, reads.index.to_numpy())
    aln, rid = aln[ok], rid[ok]
    r = reads.loc[rid]
    np.savez(kdir / "molecules" / f"chunk_{k}.npz",
             read=rid, cell=r["cell"].to_numpy(np.int32),
             umi=encode_2bit(r["umi"].to_numpy(), 10),
             half=(r["primer"] == "random hexamer").to_numpy(np.uint8),
             cdna_len=r["cdna_len"].to_numpy(np.int32),
             tx_idx=pos[aln["tx"].astype(str)].to_numpy(np.int32),
             tstart=aln["tstart"].to_numpy(np.int32), tend=aln["tend"].to_numpy(np.int32),
             tlen=aln["tlen"].to_numpy(np.int32))
    print(f"chunk {k}: {len(np.unique(rid))} reads in cells, {len(rid)} alignments", flush=True)


def combine(kdir, ref_dir):
    parts = [dict(np.load(f)) for f in sorted((kdir / "molecules").glob("chunk_*.npz"))]
    a = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    # one read per molecule: longest cDNA among reads with the same (cell, UMI, primer)
    rd = pd.DataFrame({"read": a["read"], "cell": a["cell"], "umi": a["umi"], "half": a["half"],
                       "cdna_len": a["cdna_len"]}).drop_duplicates("read")
    rd = rd.sort_values("cdna_len", ascending=False).drop_duplicates(["cell", "umi", "half"])
    mol_of = pd.Series(np.arange(len(rd)), index=rd["read"].to_numpy())
    keep = np.isin(a["read"], rd["read"].to_numpy())
    mol = mol_of[a["read"][keep]].to_numpy()
    out = {"mol": mol, "cell": a["cell"][keep], "half": a["half"][keep]}
    for key in ["tx_idx", "tstart", "tend", "tlen"]:
        out[key] = a[key][keep]
    np.savez(kdir / "molecules" / "alignments.npz", **out)

    # gene of each molecule if its alignments share one gene
    names, _ = read_fasta_lengths(ref_dir / "spliceu.fa")
    t2g = pd.read_csv(ref_dir / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    gcode, gnames = pd.factorize(t2g.reindex(names))
    ga = gcode[out["tx_idx"]]
    n_mol = int(out["mol"].max()) + 1
    gmin = np.full(n_mol, np.iinfo(np.int64).max); gmax = np.full(n_mol, -1)
    np.minimum.at(gmin, out["mol"], ga); np.maximum.at(gmax, out["mol"], ga)
    uniq = np.flatnonzero((gmin == gmax) & (gmin >= 0))
    mcell_all = np.zeros(n_mol, dtype=np.int64); mcell_all[out["mol"]] = out["cell"]
    g = pd.Series(np.asarray(gnames)[gmin[uniq]], index=uniq)
    mcell = mcell_all[uniq]
    sym = {}
    with open(ref_dir / "basic.gtf") as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) > 8 and f[2] == "gene":
                sym[re.search('gene_id "([^"]+)"', f[8]).group(1)] = re.search('gene_name "([^"]+)"', f[8]).group(1)
    genes, gidx = np.unique(g.to_numpy().astype(str), return_inverse=True)
    n_cells = int(out["cell"].max()) + 1
    M = sp.csr_matrix((np.ones(len(gidx)), (mcell.astype(np.int64), gidx)), shape=(n_cells, len(genes)))
    scipy.io.mmwrite(kdir / "molecules" / "counts.mtx", M)
    cells = pd.read_csv(kdir / "cells.tsv", sep="\t")
    (kdir / "molecules" / "cells.txt").write_text("\n".join(pd.unique(cells["cell"])) + "\n")
    (kdir / "molecules" / "genes.txt").write_text("\n".join(sym.get(x, x) for x in genes) + "\n")
    print(f"{len(rd)} molecules, {keep.sum()} alignments; {len(g)} gene-unique molecules, "
          f"median per cell {np.median(np.asarray(M.sum(1)).ravel()):.0f}", flush=True)


if __name__ == "__main__":
    kdir, ref_dir, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    (kdir / "molecules").mkdir(exist_ok=True)
    if stage == "chunk":
        chunk(kdir, ref_dir, sys.argv[4])
    else:
        combine(kdir, ref_dir)
