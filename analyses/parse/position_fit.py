"""
Primer-specific read-position models on Parse pseudobulks.

Per (subject, cell type) with at least --min-cells cells, the poly(dT) and
random-hexamer halves form two libraries of one unit. Fits:
  dT, hex                      each primer alone, no position model
  dT+pos, hex+pos              each primer alone with its position model
                               (dT: distance to the 3' end; hexamer: relative position)
  dT+hex, dT+hex+pos           both primers jointly, without and with position models
Both primers sample the same molecules, so a position model that removes
primer-specific distortion should bring the dT-only and hexamer-only isoform
fractions closer. Agreement is scored on spliced transcripts with
evaluate.concordance_metrics.

Run: python analyses/parse/position_fit.py <series_dir> <ref_dir> <out_dir> [fraction] [min_cells]
    series_dir: libraries.tsv, cells.tsv, <library>/rad_dump (make_libraries.py, dump_positions.sh)
    ref_dir:    spliceu.fa (reference order of the RAD files) and t2g.tsv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.evaluate import concordance_metrics, read_t2g
from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import FeatureColumns, ReadPositionModel, StackedFeatures, fit_bias
from joli.paf import read_fasta_lengths
from joli.rad import libraries_from_molecules, load_rad_dump, molecule_alignments

READ_LEN = 150
PRIMERS = {"poly(dT)": ("dT", "3p"), "random hexamer": ("hex", "relative")}
COLUMN = {"poly(dT)": 0, "random hexamer": 1}      # column of each primer in the stacked features


def main(series_dir, ref_dir, out_dir, fraction=1.0, min_cells=200):
    series_dir, ref_dir, out_dir = Path(series_dir), Path(ref_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, tx_len = read_fasta_lengths(ref_dir / "spliceu.fa")
    t2g = read_t2g(ref_dir / "t2g.tsv")
    cells = pd.read_csv(series_dir / "cells.tsv", sep="\t")
    libs = pd.read_csv(series_dir / "libraries.tsv", sep="\t")["library"]
    span = np.maximum(tx_len - READ_LEN, 0) + 1.0          # positions a read can start at
    rates = {"poly(dT)": np.ones(len(names)), "random hexamer": span}

    # molecules of all libraries, cells indexed globally
    parts, offset = [], 0
    for lib in libs:
        t0 = time.time()
        dump = load_rad_dump(series_dir / lib / "rad_dump")
        if dump["refs"] != names:
            raise ValueError(f"{lib}: RAD reference order differs from {ref_dir / 'spliceu.fa'}")
        c = cells[cells["library"] == lib]
        a = molecule_alignments(dump, pd.Series(c.index.to_numpy(), index=c["barcode"]), fraction=float(fraction))
        a["read"] += offset
        offset = int(a["read"].max()) + 1 if len(a) else offset
        parts.append(a)
        print(f"{lib}: {a['read'].nunique()} molecules, {len(a)} alignments ({time.time() - t0:.0f} s)", flush=True)
    aln = pd.concat(parts, ignore_index=True)
    del parts

    cells["unit"] = cells["subject"] + "|" + cells["cell_type"]
    n_cells = cells.groupby("unit").size() // 2
    units = sorted(n_cells[n_cells >= int(min_cells)].index)
    cells["group"] = cells["unit"] + "|" + cells["primer"]
    keep_cells = cells["unit"].isin(units).to_numpy()
    aln = aln[keep_cells[aln["cell"].to_numpy()]]
    group_rates = {g: rates[g.rsplit("|", 1)[1]] for g in cells["group"].unique()}
    models = {p: ReadPositionModel(tx_len, READ_LEN, mode=m) for p, (_, m) in PRIMERS.items()}
    feat_model = StackedFeatures(models.values())
    lib_of = libraries_from_molecules(aln, cells["group"].to_numpy(), names, group_rates, model=feat_model)
    print(f"{len(units)} units, {len(lib_of)} libraries", flush=True)

    tpm, summary, profiles = {}, [], []
    for unit in units:
        L = {p: lib_of[f"{unit}|{p}"] for p in PRIMERS if f"{unit}|{p}" in lib_of}
        if len(L) < 2:
            continue
        fits = {}
        for p, (short, mode) in PRIMERS.items():
            fits[short] = ([L[p]], {})
            fits[f"{short}+pos"] = ([L[p]], {0: FeatureColumns(ReadPositionModel(tx_len, READ_LEN, mode=mode), [COLUMN[p]])})
        both = [L["poly(dT)"], L["random hexamer"]]
        fits["dT+hex"] = (both, {})
        fits["dT+hex+pos"] = (both, {0: FeatureColumns(ReadPositionModel(tx_len, READ_LEN, mode="3p"), [0]),
                                     1: FeatureColumns(ReadPositionModel(tx_len, READ_LEN, mode="relative"), [1])})
        for fit, (libs_f, pos) in fits.items():
            t0 = time.time()
            em = JointEM(libs_f, units=[0] * len(libs_f))
            if pos:
                res = fit_bias(em, trunc=pos, n_outer=8)
            else:
                for l in range(len(libs_f)):       # features unused: c = 1
                    em.set_bias(l, c=np.ones_like(em.libs[l].c))
                res = em.run()
            tpm[(unit, fit)] = molar_to_tpm(res.theta[0])
            for l, m in pos.items():
                profiles.append(pd.DataFrame({"unit": unit, "fit": fit, "library": libs_f[l].name,
                                              "bin_lo": m.model.prof.edges[:-1], "p": m.model.prof.p}))
            summary.append({"unit": unit, "fit": fit, "loglik": res.loglik[-1], "n_iter": res.n_iter,
                            "molecules": sum(float(x.tcc.ec_counts.sum()) for x in libs_f),
                            "seconds": round(time.time() - t0, 1)})
            print(summary[-1], flush=True)

    spliced = np.array([not n.endswith("-U") for n in names])
    rows = []
    for unit in units:
        if (unit, "dT") not in tpm:
            continue
        get = lambda f: pd.Series(tpm[(unit, f)][spliced], index=np.array(names)[spliced])
        for a, b in [("dT", "hex"), ("dT+pos", "hex+pos")]:
            rows.append({"unit": unit, "a": a, "b": b, **concordance_metrics(get(a), get(b), t2g, min_gene_tpm=20)})
    conc = pd.DataFrame(rows)
    cols = ["pearson_log1p", "psi_pearson", "gene_tv_median", "major_isoform_agree"]
    print(conc.groupby(["a", "b"])[cols].mean().round(3).to_string())
    conc.to_csv(out_dir / "primer_concordance.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "fits.tsv", sep="\t", index=False)
    if profiles:
        pd.concat(profiles).to_csv(out_dir / "position_profiles.tsv", sep="\t", index=False)
    pd.DataFrame({f"{u}|{f}": v for (u, f), v in tpm.items()}, index=names).to_csv(
        out_dir / "tpm.tsv.gz", sep="\t", float_format="%.6g")


if __name__ == "__main__":
    main(*sys.argv[1:4], *(sys.argv[4:6]))
