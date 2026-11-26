"""
SCOTCH ONT reads (already demultiplexed by Parse split-pipe; the barcode end is trimmed, so analyses/igvf_cortex/
ont_barcodes.py finds no linkers): cell, UMI and RT primer from the read name
(<bc1 well>_<bc2 well>_<bc3 well>__<T/R>__<raw wells>__<bc1>_<bc2>_<bc3>__<UMI>), cell named <bc1 well>_<bc2 index>_
<bc3 index> as in prepare.py / sr_cells.py. Reads are oriented by the template-switch oligo (TSO): the cDNA after a TSO
found near the start is in sense orientation; a read with the reverse-complemented TSO near its end is reverse
complemented first; reads with neither are dropped. Writes the analyses/kinnex_parse chunk layout (reads/cdna_<k>.fa.gz,
reads/reads_<k>.tsv.gz, reads/summary_<k>.tsv) for alignment and molecule collapse (igvf_cortex/run.sh align ...).

Run: python analyses/scotch_pbmc/ont_from_headers.py <sample_dir> <v1 onlist> <fastq.gz> [chunk]
"""

import gzip
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

TSO = "AAGCAGTGGTATCAACGCAGAGTGAATGGG"
TSO_CORE = "GTATCAACGCAGAGT"
WINDOW = 200
COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(COMP)[::-1]


def main(sdir, v1_path, fastq, chunk=0):
    d, k = Path(sdir), int(chunk)
    v1 = {w.strip(): i for i, w in enumerate(gzip.open(v1_path, "rt")) if w.strip()}
    bmap = pd.read_csv(d / "bc1_map.tsv", sep="\t").set_index("barcode")
    well = bmap["well"].to_dict()
    sample = bmap["sample description"].iloc[0]
    rd = d / "reads"
    rd.mkdir(exist_ok=True)
    status = Counter()
    core_rc = rc(TSO_CORE)
    with gzip.open(fastq, "rt") as fh, gzip.open(rd / f"cdna_{k:03d}.fa.gz", "wt", compresslevel=1) as fa, \
            gzip.open(rd / f"reads_{k:03d}.tsv.gz", "wt", compresslevel=1) as tsv:
        tsv.write("read\tcb\tumi\tprimer\tmouse\tcdna_len\tpoly_t\ttso\n")
        i = 0
        while True:
            head = fh.readline()
            if not head:
                break
            seq = fh.readline().strip()
            fh.readline(); fh.readline()
            rid = k * 1_000_000_000 + i
            i += 1
            fields = head.split()
            try:
                f = fields[1].split("__")
                b1, b2, b3 = f[3].split("_")
                umi, primer = f[4][:10], f[1]
                cb = f"{well[b1]}_{v1[b2]}_{v1[b3]}"
            except (IndexError, KeyError, ValueError):
                status["no barcodes"] += 1
                continue
            p = seq.find(TSO_CORE, 0, WINDOW)
            if p >= 0:
                start = seq.find("GGG", p + len(TSO_CORE), p + len(TSO_CORE) + 20)
                cdna = seq[(start + 3) if start >= 0 else p + len(TSO_CORE):]
            else:
                q = seq.rfind(core_rc, max(0, len(seq) - WINDOW))
                if q < 0:
                    status["no TSO"] += 1
                    continue
                cdna = rc(seq[:q])
            # trailing poly(A) of the sense cDNA
            j = len(cdna)
            while j > 0 and cdna[j - 1] == "A":
                j -= 1
            poly_a = len(cdna) - j
            cdna = cdna[:j] if poly_a >= 8 else cdna
            if len(cdna) < 50:
                status["short"] += 1
                continue
            status["ok"] += 1
            fa.write(f">{rid}\n{cdna}\n")
            tsv.write(f"{rid}\t{cb}\t{umi}\t{'poly(dT)' if primer == 'T' else 'random hexamer'}\t{sample}\t{len(cdna)}\t"
                      f"{poly_a}\t1\n")
    s = pd.Series(status).rename("reads")
    s.to_csv(rd / f"summary_{k:03d}.tsv", sep="\t")
    print(s.to_string(), (s / s.sum()).round(3).to_string(), sep="\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
