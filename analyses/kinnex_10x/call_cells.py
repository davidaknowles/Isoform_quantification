"""
Cell calling for the Kinnex 10x PBMC long reads (extract.awk read tables).

PacBio's reads are already one per molecule with corrected barcodes; a cell is
a library-prefixed 10x barcode whose molecule count passes the barcode-rank
inflection of its library (cells.knee_threshold), so all seven libraries are
called the same way (two lack PacBio's own cell calls).

Writes <out_dir>/cells.tsv (barcode, cell, primer, kit, library, umis) and
barcode_ranks.tsv (molecules per barcode, by library).

Run: python analyses/kinnex_10x/call_cells.py <out_dir>
    out_dir holds reads/reads_<lib>_<k>.tsv.gz
"""

import sys
from pathlib import Path

import pandas as pd

from joli.cells import knee_threshold


def main(out_dir):
    out_dir = Path(out_dir)
    counts, kits = [], {}
    for f in sorted((out_dir / "reads").glob("reads_*.tsv.gz")):
        for ch in pd.read_csv(f, sep="\t", usecols=["cb", "kit"], chunksize=5_000_000):
            counts.append(ch["cb"].value_counts())
            kits.update(dict(zip(ch["cb"].str.split("_").str[0], ch["kit"])))
        print(f"{f.name}", flush=True)
    umis = pd.concat(counts).groupby(level=0).sum()
    lib = umis.index.str.split("_").str[0]
    rows, ranks = [], []
    for l in sorted(set(lib)):
        u = umis[lib == l]
        thr = knee_threshold(u.to_numpy())
        keep = u[u >= thr]
        rows.append(pd.DataFrame({"barcode": keep.index, "cell": keep.index, "primer": "poly(dT)",
                                  "kit": kits[l], "library": l, "umis": keep.to_numpy()}))
        ranks.append(pd.DataFrame({"library": l, "umis": u.sort_values(ascending=False).to_numpy()}))
        print(f"{l} ({kits[l]}): {len(u)} barcodes, threshold {thr:.0f}, {len(keep)} cells, "
              f"median molecules {keep.median():.0f}, molecules in cells {keep.sum() / u.sum():.3f}", flush=True)
    pd.concat(rows).to_csv(out_dir / "cells.tsv", sep="\t", index=False)
    pd.concat(ranks).to_csv(out_dir / "barcode_ranks.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
