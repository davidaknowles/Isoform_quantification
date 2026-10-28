"""
Library and cell tables for running salmon/alevin-fry on a public Parse
series, from a tealeaf recipe (recipe.json: libraries with their sequencing
runs and labelled cell barcodes, each barcode tagged with subject, cell type
and RT primer, poly(dT) or random hexamer).

Writes to <out_dir>:
  libraries.tsv           library, comma-separated read-1 and read-2 FASTQs
  <library>/valid_bc.txt  the library's labelled barcodes (alevin-fry --valid-bc)
  cells.tsv               barcode, library, subject, cell_type, primer

Barcodes are bc3+bc2+bc1 as read from read 2, the order salmon's
--splitseqV2 geometry concatenates them.

Run: python analyses/parse/make_libraries.py <recipe.json> <series_data_dir> <out_dir>
    series_data_dir holds manifest/ena_fastqs.tsv and fastq/<batch>/<run>_{1,2}.fastq.gz
"""

import json
import sys
from pathlib import Path

import pandas as pd


def main(recipe_path, data_dir, out_dir):
    recipe = json.loads(Path(recipe_path).read_text())
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(data_dir / "manifest" / "ena_fastqs.tsv", sep="\t", dtype=str)
    batch_of = manifest.drop_duplicates("run_accession").set_index("run_accession")["batch"]
    libs, cells = [], []
    for lib in recipe["bams"]:
        name = lib["library"]
        runs = [Path(r["path"]).parent.name for r in lib["runs"]]
        fq = {m: [str(data_dir / "fastq" / batch_of[r] / f"{r}_{m}.fastq.gz") for r in runs] for m in (1, 2)}
        missing = [f for f in fq[1] + fq[2] if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"{name}: {missing[:3]}")
        libs.append({"library": name, "read1": ",".join(fq[1]), "read2": ",".join(fq[2])})
        (out_dir / name).mkdir(exist_ok=True)
        (out_dir / name / "valid_bc.txt").write_text("\n".join(lib["barcode_groups"]) + "\n")
        for bc, (subject, cell_type, primer) in lib["barcode_groups"].items():
            cells.append({"barcode": bc, "library": name, "subject": subject,
                          "cell_type": cell_type, "primer": primer})
    pd.DataFrame(libs).to_csv(out_dir / "libraries.tsv", sep="\t", index=False)
    cells = pd.DataFrame(cells)
    cells.to_csv(out_dir / "cells.tsv", sep="\t", index=False)
    print(cells.groupby(["library", "primer"]).size().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
