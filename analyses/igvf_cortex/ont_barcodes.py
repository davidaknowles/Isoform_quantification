"""
Parse Evercode WT v2 cell barcodes, UMIs and RT primers in ONT reads of the IGVF B01_13H cortex sublibrary, with
tealeaf's long-read Parse caller (tealeaf.sc.parse_longread._call: linkers placed by semi-global edit distance in both
strands, barcodes corrected to their whitelists) given the WT v2 linkers.

WT v2 molecule in read-2 orientation (as Illumina read 2 reads it; the seqspec linker and onlist sequences are given in
this orientation, checked on read 2: linker-1 at 56-78, bc1 at 78-86, exact whitelist matches 73-84%):
    UMI (10) | bc3 (8) | linker-2 (30) | bc2 (8) | linker-1 (22) | bc1 (8) | poly(T) or hexamer | cDNA | TSO
Whitelists: rounds 2 and 3 the v1 list (IGVF onlist IGVFFI1138MCVX), round 1 the subpool's barcode map (IGVFFI9171FPFR:
bc1 barcode -> well, RT primer T/R, mouse). Cells are named <bc1 well>_<bc2 index>_<bc3 index>, so a well's oligo-dT and
random-hexamer barcodes give one cell (as the IGVF barcode replacement list merges them).

One chunk per input FASTQ (--chunk k), in the layout of analyses/kinnex_parse (molecules.py then joins alignments
to cells and collapses molecules): <out>/reads/cdna_<k>.fa.gz (cDNA inserts in sense orientation, named by integer
read id k * 1e9 + i) and <out>/reads/reads_<k>.tsv.gz (read, cb, umi, primer "poly(dT)" / "random hexamer", mouse,
cdna_len, poly_t, tso), with <out>/reads/summary_<k>.tsv (reads by calling status). Calling rate on 20,000 reads:
38% (44% of reads have no linker in either end, probably lacking the barcode end; a wider window or more linker edits
recover < 2% more).

Needs numba and tealeaf (~/venv/kinnex).

Run: ~/venv/kinnex/bin/python analyses/igvf_cortex/ont_barcodes.py <igvf_dir> <out_dir> <fastq.gz> --chunk <k> [--limit N]
"""

import argparse
import gzip
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/commons/home/daknowles/projects/tealeaf")
from tealeaf.sc import parse_longread as pl  # noqa: E402
from tealeaf.sc.native_align import CODE, encode  # noqa: E402

LINKER_1 = "ATCCACGTGCTTGAGACTGTGG"              # read-2 orientation, between bc2 and bc1
LINKER_2 = "GTGGCCGATGTTTCGCATCGGCGTACGACT"      # read-2 orientation, between bc3 and bc2


def read_fastq(paths, limit=0):
    n = 0
    for p in paths:
        with gzip.open(p, "rt") as fh:
            while True:
                head = fh.readline()
                if not head:
                    break
                seq, _, qual = fh.readline().strip(), fh.readline(), fh.readline().strip()
                fields = head[1:].split()
                yield (fields[0] if fields else f"unnamed_{n}"), seq, qual
                n += 1
                if limit and n >= limit:
                    return


def call(seqs, bc3, bc2, bc1, window, max_linker_edits, max_edits):
    joined = np.frombuffer("".join(seqs).encode(), dtype=np.uint8)
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s in seqs])
    out = np.zeros((len(seqs), len(pl.COLUMNS)), dtype=np.int64)
    pl._call(CODE[joined], offsets, encode(LINKER_2), encode(LINKER_1),
             encode(pl.reverse_complement(pl.TSO_CORE)), pl.word_codes(bc3), pl.word_codes(bc2), pl.word_codes(bc1),
             window, max_linker_edits, max_edits, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("igvf_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("fastq", nargs="+")
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--max-linker-edits", type=int, default=5)
    ap.add_argument("--max-edits", type=int, default=1)
    ap.add_argument("--rc-whitelists", action="store_true", help="reverse-complement the onlists (orientation check)")
    ap.add_argument("--bmap", type=Path, help="round-1 table (barcode, sample description, parse barcode type, well); "
                    "default <igvf_dir>/meta/IGVFFI9171FPFR.tsv.gz")
    ap.add_argument("--v1", type=Path, help="round-2/3 onlist; default <igvf_dir>/meta/IGVFFI1138MCVX.txt.gz")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    meta = a.igvf_dir / "meta"
    v1 = [w.strip() for w in gzip.open(a.v1 or meta / "IGVFFI1138MCVX.txt.gz", "rt") if w.strip()]
    bmap = pd.read_csv(a.bmap or meta / "IGVFFI9171FPFR.tsv.gz", sep="\t")
    bc1 = bmap["barcode"].tolist()
    if a.rc_whitelists:
        v1 = [pl.reverse_complement(w) for w in v1]
        bc1 = [pl.reverse_complement(w) for w in bc1]
    well, primer, mouse = bmap["well"].tolist(), bmap["parse barcode type"].tolist(), bmap["sample description"].tolist()
    status = Counter()
    rd = a.out_dir / "reads"
    rd.mkdir(parents=True, exist_ok=True)
    k = a.chunk
    primer_name = {"T": "poly(dT)", "R": "random hexamer"}
    n_read = 0
    with gzip.open(rd / f"cdna_{k:03d}.fa.gz", "wt", compresslevel=1) as fa, \
            gzip.open(rd / f"reads_{k:03d}.tsv.gz", "wt", compresslevel=1) as tsv:
        tsv.write("read\tcb\tumi\tprimer\tmouse\tcdna_len\tpoly_t\ttso\n")

        def flush(batch):
            nonlocal n_read
            rows = call([s for _, s, _ in batch], v1, v1, bc1, a.window, a.max_linker_edits, a.max_edits)
            for (name, seq, qual), row in zip(batch, rows):
                rid = k * 1_000_000_000 + n_read
                n_read += 1
                status[pl.STATUS[row[0]]] += 1
                if row[0] != 0:
                    continue
                umi, insert, _ = pl.extract(seq, qual, row)
                if not insert:
                    continue
                i3, i2, i1 = row[2], row[3], row[4]
                fa.write(f">{rid}\n{insert}\n")
                tsv.write(f"{rid}\t{well[i1]}_{i2}_{i3}\t{umi}\t{primer_name[primer[i1]]}\t{mouse[i1]}\t{len(insert)}\t"
                          f"{row[11]}\t{row[12]}\n")

        batch = []
        for rec in read_fastq(a.fastq, a.limit):
            batch.append(rec)
            if len(batch) >= a.batch:
                flush(batch)
                batch = []
        if batch:
            flush(batch)
    s = pd.Series(status).rename("reads")
    s.to_csv(rd / f"summary_{k:03d}.tsv", sep="\t")
    print(s.to_string(), (s / s.sum()).round(3).to_string(), sep="\n")


if __name__ == "__main__":
    main()
