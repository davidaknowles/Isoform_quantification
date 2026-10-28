"""
Call cell barcodes, UMIs and RT primers on PacBio Kinnex Parse S-reads and
write the trimmed sense cDNA for alignment.

Reads FASTQ on stdin (e.g. `samtools fastq segmented.bam`). Outputs in out_dir,
in chunks of --chunk reads:
  cdna_<k>.fa.gz       cDNA, named by the S-read's index in the input
  reads_<k>.tsv.gz     read, cb (bc3+bc2+bc1), umi, primer, polya, tso, n_corrected, cdna_len
  stats.json           counts per outcome

Whitelists and the bc1 -> primer map come from a tealeaf recipe of a Parse WT
v3 series (barcode_segment_{1,2,3}.txt and recipe.json barcode tags).

Run: samtools fastq -@4 segmented.bam | python analyses/kinnex_parse/extract.py <recipe_dir> <out_dir> [threads] [chunk]
"""

import gzip
import json
import sys
from collections import Counter
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

from joli.parse_lr import ParseReadParser

_parser = None


def _init(recipe_dir):
    global _parser
    recipe_dir = Path(recipe_dir)
    wl = [(recipe_dir / "whitelists" / f"barcode_segment_{i}.txt").read_text().split() for i in (1, 2, 3)]
    recipe = json.loads((recipe_dir / "recipe.json").read_text())
    prim = {bc[16:]: p for lib in recipe["bams"] for bc, (_, _, p) in lib["barcode_groups"].items()}
    _parser = ParseReadParser(wl, prim)


def _work(batch):
    out, why = [], Counter()
    for idx, seq in batch:
        r, reason = _parser.parse(seq)
        why[reason] += 1
        if r is not None:
            out.append((idx, r.bc3 + r.bc2 + r.bc1, r.umi, r.primer, r.polya, int(r.tso), r.n_corrected, r.cdna))
    return out, why


def _batches(stream, size=20000):
    idx = 0
    while True:
        lines = list(islice(stream, 4 * size))
        if not lines:
            return
        seqs = lines[1::4]
        yield [(idx + i, s.rstrip("\n")) for i, s in enumerate(seqs)]
        idx += len(seqs)


def main(recipe_dir, out_dir, threads=8, chunk=10_000_000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threads, chunk = int(threads), int(chunk)
    stats, k, n_in_chunk = Counter(), 0, 0
    fa = tsv = None

    def open_chunk(k):
        f = gzip.open(out_dir / f"cdna_{k:03d}.fa.gz", "wt", compresslevel=1)
        t = gzip.open(out_dir / f"reads_{k:03d}.tsv.gz", "wt", compresslevel=1)
        t.write("read\tcb\tumi\tprimer\tpolya\ttso\tn_corrected\tcdna_len\n")
        return f, t

    with Pool(threads, initializer=_init, initargs=(recipe_dir,)) as pool:
        for out, why in pool.imap(_work, _batches(sys.stdin), chunksize=1):
            stats.update(why)
            for idx, cb, umi, primer, polya, tso, nc, cdna in out:
                if fa is None or n_in_chunk >= chunk:
                    if fa is not None:
                        fa.close(); tsv.close(); k += 1
                    fa, tsv = open_chunk(k)
                    n_in_chunk = 0
                fa.write(f">{idx}\n{cdna}\n")
                tsv.write(f"{idx}\t{cb}\t{umi}\t{primer}\t{polya}\t{tso}\t{nc}\t{len(cdna)}\n")
                n_in_chunk += 1
    if fa is not None:
        fa.close(); tsv.close()
    (out_dir / "stats.json").write_text(json.dumps(dict(stats), indent=1))
    print(dict(stats), file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:5])
