"""
Libraries for the mouse-brain pilot, cached for fitting.

Short reads: one Library per Tabula Muris Senis pseudobulk group (make_groups.py, sr_group.sh) from salmon
fragment mappings, with FragmentModel features (fragment-length and relative-start bins) and effective-
length rates; an optional random fraction of fragments. Long reads: one Library per ENCODE PacBio fastq
(lr_align.sh) from minimap2 best-score ties, with AnchoredTruncationModel features and poly(dT) rates
(1 on spliced targets, KAPPA0 (L + 1) on unspliced gene-span targets).

Writes <out.pkl>: reference (names, lengths, unspliced mask, gene codes, gene groups with unspliced targets
as groups of their own), SR libraries with initial fragment-length distributions, LR libraries, metadata.

Run: python analyses/mouse_brain/build_libraries.py <mouse_brain_dir> <ref_dir> <out.pkl> [sr_fraction]
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.alevin import primer_rates
from joli.lr_bias import AnchoredTruncationModel, FragmentModel
from joli.paf import filter_alignments, library_from_alignments, read_fasta_lengths, read_paf
from joli.salmon_io import library_from_fragments, read_fragments

KAPPA0 = np.exp(-8.6)       # poly(dT) internal priming per base on pre-mRNA (as for the Parse/Kinnex fits)


def main(d, ref, out, sr_fraction=1.0):
    d, ref = Path(d), Path(ref)
    names, tx_len = read_fasta_lengths(ref / "spliceu.fa")
    unspliced = np.array([n.endswith("-U") for n in names])
    t2g = pd.read_csv(ref / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    genes = pd.factorize(t2g.reindex(names))[0]
    n_genes = int(genes.max()) + 1
    gene_groups = np.where(genes < 0, -1, genes + np.where(unspliced, n_genes, 0))
    rng = np.random.default_rng(0)
    groups = pd.read_csv(d / "sr" / "groups.tsv", sep="\t")
    sr = {}
    for g in groups.itertuples():
        gd = d / "sr" / g.group.replace("|", "_")
        if not (gd / "fragments.tsv.gz").exists():
            print(f"missing {g.group}", flush=True)
            continue
        t0 = time.time()
        frags = read_fragments(gd / "fragments.tsv.gz", gd / "refs.txt", names, fraction=sr_fraction)
        fm = FragmentModel(tx_len)
        sr[g.group] = (library_from_fragments(f"{g.group}|SR", frags, names, tx_len, fm), fm.pf.copy())
        print(f"SR {g.group}: {frags['read'].nunique()} fragments, {len(sr[g.group][0].tcc.ec_counts)} ECs "
              f"({time.time() - t0:.0f} s)", flush=True)
        del frags
    files = pd.read_csv(d / "lr" / "files.tsv", sep="\t")
    rates = primer_rates("poly(dT)", tx_len, 0, unspliced, kappa=KAPPA0)
    feat_model = AnchoredTruncationModel(tx_len, uniform=unspliced)
    lr = {}
    for f in files.itertuples():
        paf = d / "lr" / "minimap2" / f"{f.accession}.paf.gz"
        t0 = time.time()
        aln = filter_alignments(read_paf(paf))
        feat = feat_model.features(aln["tstart"], aln["tend"], aln["tlen"])
        lr[f.accession] = library_from_alignments(f"{f.tissue}|LR|{f.accession}", aln, names, features=feat, b=rates)
        print(f"LR {f.accession} ({f.tissue}): {aln['read'].nunique()} reads ({time.time() - t0:.0f} s)", flush=True)
    with open(out, "wb") as fh:
        pickle.dump({"names": names, "tx_len": tx_len, "unspliced": unspliced, "genes": genes,
                     "gene_groups": gene_groups, "groups": groups, "files": files, "sr": sr, "lr": lr,
                     "sr_fraction": sr_fraction}, fh, protocol=4)


if __name__ == "__main__":
    main(*sys.argv[1:4], *(float(x) for x in sys.argv[4:5]))
