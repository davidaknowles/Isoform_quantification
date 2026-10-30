"""
Cell calling for the Kinnex 10x PBMC long reads (extract.awk read tables).

PacBio's reads are already one per molecule with corrected barcodes, and the
deduplicated BAMs hold only PacBio's called cells (e.g. 20,426 barcodes for the
20,426 cells of the 20k-cell 3' library). A barcode-rank inflection then cuts
inside the cell plateau, so cells are the library-prefixed barcodes with at
least min_molecules molecules.

Writes <out_dir>/cells.tsv (barcode, cell, primer, kit, library, umis) and
barcode_ranks.tsv (molecules per barcode, by library).

Run: python analyses/kinnex_10x/call_cells.py <out_dir> [min_molecules]
    out_dir holds reads/reads_<lib>_<k>.tsv.gz
"""

import sys
from pathlib import Path

import pandas as pd


def main(out_dir, min_molecules=500):
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
        thr = float(min_molecules)
        keep = u[u >= thr]
        rows.append(pd.DataFrame({"barcode": keep.index, "cell": keep.index, "primer": "poly(dT)",
                                  "kit": kits[l], "library": l, "umis": keep.to_numpy()}))
        ranks.append(pd.DataFrame({"library": l, "umis": u.sort_values(ascending=False).to_numpy()}))
        print(f"{l} ({kits[l]}): {len(u)} barcodes, threshold {thr:.0f}, {len(keep)} cells, "
              f"median molecules {keep.median():.0f}, molecules in cells {keep.sum() / u.sum():.3f}", flush=True)
    pd.concat(rows).to_csv(out_dir / "cells.tsv", sep="\t", index=False)
    pd.concat(ranks).to_csv(out_dir / "barcode_ranks.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:3])
