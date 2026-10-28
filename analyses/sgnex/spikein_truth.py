"""
Spike-in truth table for SG-NEx samples: relative molarity per spike-in
transcript for each spike-in mix.

  sequinA_v2 : RNA sequins Mix A version 2 (HEYA8), concentration column MXA
  sirv_E2    : SIRV Set 1 isoform mix E2 (HEYA8), 69 isoforms at 1/32 to 4 relative molarity
  sirv4      : SIRV Set 4 (H9), 69 E0 isoforms and 15 long SIRVs, all at 0.6 fmol

Run: python analyses/sgnex/spikein_truth.py <ref_dir> <out.tsv>
    ref_dir holds RNAsequins_MixA.xlsx and the Lexogen SIRV Set 1 design sheet
"""

import sys
from pathlib import Path

import pandas as pd


def main(ref_dir, out_path):
    ref_dir = Path(ref_dir)
    rows = []
    seq = pd.read_excel(ref_dir / "RNAsequins_MixA.xlsx", header=None).iloc[2:, [3, 4]].dropna()
    seq.columns = ["transcript", "conc"]
    seq = seq[pd.to_numeric(seq["conc"], errors="coerce").notna()]
    for t, c in zip(seq["transcript"], seq["conc"].astype(float)):
        rows.append(("sequinA_v2", t, t.rsplit("_", 1)[0], c))
    sirv = pd.read_excel(ref_dir / "SIRV_Set1_Norm_sequence-design-overview_20210507a.xlsx", header=None)
    sirv = sirv.iloc[5:, [6, 10, 12]]
    sirv.columns = ["transcript", "E0", "E2"]
    sirv = sirv[sirv["transcript"].astype(str).str.match(r"^SIRV\d{3}$") & sirv["E0"].notna()]
    for t, e2 in zip(sirv["transcript"], sirv["E2"].astype(float)):
        gene = t[:5]
        rows.append(("sirv_E2", t, gene, e2))
        rows.append(("sirv4", t, gene, 1.0))
    for g in [4, 6, 8, 10, 12]:
        for k in (1, 2, 3):
            rows.append(("sirv4", f"SIRV{g}00{k}", f"SIRV{g}00{k}", 1.0))
    out = pd.DataFrame(rows, columns=["mix", "transcript", "gene", "rel_molarity"])
    print(out.groupby("mix").agg(n=("transcript", "size"), genes=("gene", "nunique")))
    out.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
