"""
Short-read "series" directory for analyses/parse/joint_celltype.py and heldout_lr.py from the IGVF B01_13H tables:
<igvf_dir>/sr_series/libraries.tsv (one library, sr_quant), cells.tsv (SR barcode, library, subject = mouse,
cell_type from celltypist.tsv by cell, primer) and sr_series/sr_quant -> ../sr_quant (rad_dump). The long-read side
is <igvf_dir> itself (molecules/alignments.npz, molecules/cells.txt, celltypist.tsv, cells.tsv).

Run: python analyses/igvf_cortex/make_series.py <igvf_dir>
"""

import sys
from pathlib import Path

import pandas as pd


def main(igvf_dir):
    d = Path(igvf_dir)
    s = d / "sr_series"
    s.mkdir(exist_ok=True)
    if not (s / "sr_quant").exists():
        (s / "sr_quant").symlink_to("../sr_quant")
    bc = pd.read_csv(d / "sr_barcodes.tsv", sep="\t")
    ct = pd.read_csv(d / "celltypist.tsv", sep="\t").set_index("cell")["cell_type"]
    bc["cell_type"] = ct.reindex(bc["cell"]).to_numpy()
    bc = bc.dropna(subset=["cell_type"])
    pd.DataFrame({"library": ["sr_quant"]}).to_csv(s / "libraries.tsv", sep="\t", index=False)
    bc.assign(library="sr_quant", subject=bc["mouse"])[["barcode", "library", "subject", "cell_type", "primer"]].to_csv(
        s / "cells.tsv", sep="\t", index=False)
    print(bc.drop_duplicates("cell")["cell_type"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1])
