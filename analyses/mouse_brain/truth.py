"""
Single-cell long-read reference for the mouse-brain pilot: Joglekar et al. 2024 ScISOr-Seq2 annotated-transcript
UMI counts (P56) aggregated to the regions and coarse cell classes of the short-read groups.

Source columns are age::mouse::region::lineage::type::subtype. Regions: Hippocampus, VisCortex (-> Cortex),
Striatum, Cerebellum (Thalamus has no short-read counterpart). Subtypes -> classes: astrocyte (Astrocytes,
BergmannGlia), oligodendrocyte (MFOLs, MOLs), OPC (OPCs, DivOPCs), microglia (Microglia, Macrophages),
endothelial (Endo), pericyte (Pericytes), neuron (excitatory and inhibitory neuron subtypes); others left out.
Transcript names are mapped to transcript ids through transcript_name in the reference GTF.

Writes <out>: transcripts (rows) x "<region>|<class>|<mouse>" UMI counts, plus "<region>|<class>|all".

Run: python analyses/mouse_brain/truth.py <matrix_dir> <reference_gtf> <out.tsv.gz>
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

REGION = {"Hippocampus": "Hippocampus", "VisCortex": "Cortex", "Striatum": "Striatum", "Cerebellum": "Cerebellum"}
CLASS = {"Astrocytes": "astrocyte", "BergmannGlia": "astrocyte", "MFOLs": "oligodendrocyte", "MOLs": "oligodendrocyte",
         "OPCs": "OPC", "DivOPCs": "OPC", "Microglia": "microglia", "Macrophages": "microglia", "Endo": "endothelial",
         "Pericytes": "pericyte"}


def subtype_class(lineage, typ, sub):
    if lineage == "Neuron" and typ in ("ExciteNeuron", "InhibNeuron"):
        return "neuron"
    return CLASS.get(sub)


def main(matrix_dir, gtf, out):
    d = Path(matrix_dir)
    feat = pd.read_csv(d / "features.tsv.gz", sep="\t", header=None, names=["gene_id", "transcript_name"])
    cols = pd.read_csv(d / "barcodes.tsv.gz", sep="\t", header=None, names=["label"])["label"]
    M = mmread(d / "matrix.mtx.gz").tocsc()
    names = {}
    with open(gtf) as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) > 8 and f[2] == "transcript":
                names[re.search('transcript_name "([^"]+)"', f[8]).group(1)] = re.search('transcript_id "([^"]+)"', f[8]).group(1)
    tid = feat["transcript_name"].map(names)
    print(f"{tid.notna().mean():.3f} of source transcripts named in the reference")
    groups = {}
    for j, lab in enumerate(cols):
        age, mouse, region, lineage, typ, sub = lab.split("::")
        cls = subtype_class(lineage, typ, sub)
        if age != "P56" or region not in REGION or cls is None:
            continue
        groups.setdefault(f"{REGION[region]}|{cls}|{mouse}", []).append(j)
    ok = tid.notna().to_numpy()
    data = {g: np.asarray(M[:, js].sum(axis=1)).ravel()[ok] for g, js in groups.items()}
    df = pd.DataFrame(data, index=tid[ok].to_numpy())
    df = df.groupby(level=0).sum()
    for rc in sorted({g.rsplit("|", 1)[0] for g in df.columns}):
        df[f"{rc}|all"] = df[[c for c in df.columns if c.startswith(rc + "|") and not c.endswith("|all")]].sum(axis=1)
    total = np.asarray(M.sum(axis=0)).ravel()
    used = sum(total[js].sum() for js in groups.values())
    print(f"{len(groups)} region|class|mouse groups; UMIs kept {df[[c for c in df.columns if not c.endswith('|all')]].to_numpy().sum():.0f} of {used:.0f} in those groups")
    print(df[[c for c in df.columns if c.endswith("|all")]].sum().round(0).to_string())
    df.to_csv(out, sep="\t", float_format="%.6g")


if __name__ == "__main__":
    main(*sys.argv[1:4])
