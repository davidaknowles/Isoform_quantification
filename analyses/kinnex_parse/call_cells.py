"""
Cell calling for the Kinnex Parse long reads from extract.py output.

A cell is bc3+bc2+RT well; its poly(dT) and random-hexamer halves (bc1 pairs
inferred from the labelled cells of a WT v3 series, parse_lr.primer_pairs_from_cells)
are merged. UMIs are deduplicated per (barcode, UMI); cells are barcodes whose
merged UMI count passes the barcode-rank inflection (cells.knee_threshold).

Writes <out_dir>/cells.tsv (barcode, cell, primer, umis; one row per primer half,
cell named by its poly(dT) barcode) and barcode_ranks.tsv.

Run: python analyses/kinnex_parse/call_cells.py <reads_dir> <recipe_dir> <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from joli.cells import knee_threshold
from joli.parse_lr import primer_pairs_from_cells
from joli.rad import decode_2bit, encode_2bit


def main(reads_dir, recipe_dir, out_dir):
    reads_dir, out_dir = Path(reads_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((Path(recipe_dir) / "recipe.json").read_text())
    pairs = primer_pairs_from_cells((bc, p) for lib in recipe["bams"] for bc, (_, _, p) in lib["barcode_groups"].items())
    hex_to_dt = {h: d for d, h in pairs.items()}
    keys = []
    for f in sorted(reads_dir.glob("reads_*.tsv.gz")):
        for ch in pd.read_csv(f, sep="\t", usecols=["cb", "umi", "primer"], chunksize=5_000_000):
            # cell code: bc3+bc2+poly(dT) bc1 of the well; half flag for the primer
            bc1 = ch["cb"].str[16:]
            dt_bc1 = bc1.map(lambda b: hex_to_dt.get(b, b))
            cell = encode_2bit((ch["cb"].str[:16] + dt_bc1).to_numpy(), 24)
            umi = encode_2bit(ch["umi"].to_numpy(), 10)
            half = (ch["primer"] == "random hexamer").to_numpy().astype(np.uint64)
            ok = (cell != np.iinfo(np.uint64).max) & (umi != np.iinfo(np.uint64).max)
            keys.append(np.column_stack([cell[ok], (umi[ok] << np.uint64(1)) | half[ok]]))
        print(f"{f.name}: {sum(len(k) for k in keys)} reads", flush=True)
    k = np.unique(np.concatenate(keys), axis=0)               # unique (cell, umi, half)
    cell, half = k[:, 0], (k[:, 1] & np.uint64(1)).astype(np.int64)
    ucell, inv = np.unique(cell, return_inverse=True)
    umis = np.zeros((len(ucell), 2), dtype=np.int64)
    np.add.at(umis, (inv, half), 1)
    total = umis.sum(1)
    thr = knee_threshold(total)
    keep = total >= thr
    names = np.array(decode_2bit(ucell[keep], 24))
    rows = []
    for nm, (u_dt, u_hex) in zip(names, umis[keep]):
        rows.append((nm, nm, "poly(dT)", u_dt))
        rows.append((nm[:16] + pairs.get(nm[16:], nm[16:]), nm, "random hexamer", u_hex))
    cells = pd.DataFrame(rows, columns=["barcode", "cell", "primer", "umis"])
    cells.to_csv(out_dir / "cells.tsv", sep="\t", index=False)
    pd.DataFrame({"umis": np.sort(total)[::-1]}).to_csv(out_dir / "barcode_ranks.tsv", sep="\t", index=False)
    print(f"{len(ucell)} barcodes, threshold {thr:.0f} UMIs, {keep.sum()} cells, "
          f"median UMIs {np.median(total[keep]):.0f}, UMIs in cells {total[keep].sum() / total.sum():.3f}, "
          f"hexamer share {umis[keep, 1].sum() / total[keep].sum():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
