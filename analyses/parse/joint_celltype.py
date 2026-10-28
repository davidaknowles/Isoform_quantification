"""
Joint Parse short-read + Kinnex Parse long-read quantification by cell type.

Units are CellTypist cell types (Immune_All_Low, the vocabulary of both the
short-read recipes and the long-read labels). Libraries:
  SR  one per (series, subject, cell type, primer): Parse short-read pseudobulks
      (salmon/alevin-fry RAD positions), poly(dT) rate b = 1, hexamer b = L - r + 1
  LR  one per (cell type, primer): Kinnex Parse long-read pseudobulks, b = 1
Each library belongs to the unit of its cell type. The short- and long-read
donors differ, so a unit's abundances are shared across donors and platforms.

Bias models are shared per platform x primer across all of its libraries:
  SR position profile (poly(dT): distance to 3' end; hexamer: relative position)
  LR read-end truncation (5' and 3' uncovered bases)
  LR length bias (identifiable against SR poly(dT), whose rate is fixed at 1)

Fits: SR_plain, SR, LR_plain, LR, joint_plain, joint (all bias models).
Outputs: tpm.tsv.gz (transcripts x fit|unit), fits.tsv, bias.tsv, and
concordance.tsv (per cell type: SR-only vs LR-only isoform fractions on
spliced transcripts, without and with bias models).

Run: python analyses/parse/joint_celltype.py <sr_series_dirs,comma-separated> <kinnex_dir> <ref_dir> <out_dir>
                                             [sr_fraction] [lr_fraction] [min_cells]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.evaluate import concordance_metrics, read_t2g
from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import (FeatureColumns, LengthBias, ReadPositionModel, StackedFeatures,
                          TruncationModel, fit_bias)
from joli.paf import read_fasta_lengths
from joli.rad import libraries_from_molecules, load_rad_dump, molecule_alignments

SR_READ_LEN = 150
PRIMERS = ("poly(dT)", "random hexamer")


def load_sr(series_dirs, names, fraction):
    """Short-read molecules of all series; group label per global cell index."""
    parts, groups, mol_off, cell_off = [], [], 0, 0
    for sdir in series_dirs:
        sdir = Path(sdir)
        cells = pd.read_csv(sdir / "cells.tsv", sep="\t")
        for lib in pd.read_csv(sdir / "libraries.tsv", sep="\t")["library"]:
            t0 = time.time()
            dump = load_rad_dump(sdir / lib / "rad_dump")
            if dump["refs"] != names:
                raise ValueError(f"{sdir.name}/{lib}: RAD reference order differs from spliceu.fa")
            c = cells[cells["library"] == lib].reset_index(drop=True)
            a = molecule_alignments(dump, pd.Series(np.arange(len(c)), index=c["barcode"]), fraction=fraction)
            a["read"] += mol_off
            a["cell"] += cell_off
            mol_off = int(a["read"].max()) + 1
            cell_off += len(c)
            groups += [f"{t}|SR|{p}|{sdir.name}:{s}" for t, p, s in zip(c["cell_type"], c["primer"], c["subject"])]
            parts.append(a)
            print(f"SR {sdir.name}/{lib}: {a['read'].nunique()} molecules ({time.time() - t0:.0f} s)", flush=True)
    return pd.concat(parts, ignore_index=True), np.array(groups, dtype=object)


def load_lr(kdir, fraction, seed=0):
    """Long-read molecules; cell index = 2 * cell + primer half."""
    kdir = Path(kdir)
    z = np.load(kdir / "molecules" / "alignments.npz")
    mol = z["mol"]
    keep = np.random.default_rng(seed).random(int(mol.max()) + 1) < fraction
    sel = keep[mol]
    aln = pd.DataFrame({"read": mol[sel], "cell": 2 * z["cell"][sel].astype(np.int64) + z["half"][sel],
                        "tx_idx": z["tx_idx"][sel], "start": z["tstart"][sel], "end": z["tend"][sel],
                        "tlen": z["tlen"][sel]})
    cell_names = (kdir / "molecules" / "cells.txt").read_text().split()
    ct = pd.read_csv(kdir / "celltypist.tsv", sep="\t").set_index("cell")["cell_type"]
    types = ct.reindex(cell_names).to_numpy(dtype=object)
    groups = np.array([f"{t}|LR|{p}" if isinstance(t, str) else None for t in types for p in PRIMERS],
                      dtype=object)
    aln = aln[pd.notna(groups[aln["cell"].to_numpy()])]
    print(f"LR: {aln['read'].nunique()} molecules in labelled cells", flush=True)
    return aln, groups


def main(sr_dirs, kdir, ref_dir, out_dir, sr_fraction=0.1, lr_fraction=1.0, min_cells=100):
    ref_dir, out_dir = Path(ref_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, tx_len = read_fasta_lengths(ref_dir / "spliceu.fa")
    t2g = read_t2g(ref_dir / "t2g.tsv")
    span = np.maximum(tx_len - SR_READ_LEN, 0) + 1.0

    sr_aln, sr_groups = load_sr(sr_dirs.split(","), names, float(sr_fraction))
    lr_aln, lr_groups = load_lr(kdir, float(lr_fraction))

    # units: cell types with >= min_cells in either platform
    sr_cells = pd.Series([g.split("|")[0] for g in sr_groups]).value_counts() // 2
    lr_cells = pd.Series([g.split("|")[0] for g in lr_groups if g is not None]).value_counts() // 2
    n_cells = sr_cells.add(lr_cells, fill_value=0)
    units = sorted(n_cells[(sr_cells.reindex(n_cells.index, fill_value=0) >= int(min_cells))
                           | (lr_cells.reindex(n_cells.index, fill_value=0) >= int(min_cells))].index)
    both = [u for u in units if sr_cells.get(u, 0) >= int(min_cells) and lr_cells.get(u, 0) >= int(min_cells)]
    print(f"{len(units)} units, {len(both)} with both platforms: {both}", flush=True)
    unit_idx = {u: k for k, u in enumerate(units)}

    sr_feat = StackedFeatures([ReadPositionModel(tx_len, SR_READ_LEN, "3p"),
                               ReadPositionModel(tx_len, SR_READ_LEN, "relative")])
    rate = lambda g: span if "|SR|random hexamer|" in g else np.ones(len(names))
    keep_g = lambda gs: {g: rate(g) for g in set(x for x in gs if x is not None) if g.split("|")[0] in unit_idx}
    sr_aln = sr_aln[pd.Series(sr_groups[sr_aln["cell"].to_numpy()]).isin(keep_g(sr_groups)).to_numpy()]
    lr_aln = lr_aln[pd.Series(lr_groups[lr_aln["cell"].to_numpy()]).isin(keep_g(lr_groups)).to_numpy()]
    sr_libs = libraries_from_molecules(sr_aln, sr_groups, names, keep_g(sr_groups), model=sr_feat)
    lr_libs = libraries_from_molecules(lr_aln, lr_groups, names, keep_g(lr_groups),
                                       model=TruncationModel(tx_len), platform="long")
    del sr_aln, lr_aln
    print(f"{len(sr_libs)} SR and {len(lr_libs)} LR libraries", flush=True)

    def setup(lib_dict, use_models, sr_models, lr_models, lr_length):
        keys = sorted(lib_dict)
        libs = [lib_dict[k] for k in keys]
        unit_of = [unit_idx[k.split("|")[0]] for k in keys]
        trunc, length = {}, {}
        if use_models:
            for i, k in enumerate(keys):
                plat, primer = k.split("|")[1:3]
                if plat == "SR":
                    trunc[i] = sr_models[primer]
                else:
                    trunc[i] = lr_models[primer]
                    if lr_length is not None:
                        length[i] = lr_length[primer]
        return keys, libs, unit_of, trunc, length

    fits = {"SR_plain": (sr_libs, False, False), "SR": (sr_libs, True, False),
            "LR_plain": (lr_libs, False, False), "LR": (lr_libs, True, False),
            "joint_plain": ({**sr_libs, **lr_libs}, False, False),
            "joint": ({**sr_libs, **lr_libs}, True, True)}
    tpm, summary, bias = {}, [], []
    for fit, (lib_dict, use_models, use_length) in fits.items():
        t0 = time.time()
        sr_models = {"poly(dT)": FeatureColumns(ReadPositionModel(tx_len, SR_READ_LEN, "3p"), [0]),
                     "random hexamer": FeatureColumns(ReadPositionModel(tx_len, SR_READ_LEN, "relative"), [1])}
        lr_models = {p: TruncationModel(tx_len) for p in PRIMERS}
        lr_length = {p: LengthBias(tx_len) for p in PRIMERS} if use_length else None
        keys, libs, unit_of, trunc, length = setup(lib_dict, use_models, sr_models, lr_models, lr_length)
        present = sorted(set(unit_of))                    # units covered by this fit's libraries
        local = {k: i for i, k in enumerate(present)}
        pi = np.zeros((len(libs), len(present)))
        pi[np.arange(len(libs)), [local[k] for k in unit_of]] = 1.0
        em = JointEM(libs, pi=pi, unit_names=[units[k] for k in present])
        res = fit_bias(em, trunc=trunc, length=length, n_outer=8, verbose=True) if (trunc or length) else em.run()
        for k in present:
            tpm[f"{fit}|{units[k]}"] = molar_to_tpm(res.theta[local[k]])
        if use_models:
            for p, m in sr_models.items():
                bias.append(pd.DataFrame({"fit": fit, "model": f"SR position {p}", "x": m.model.prof.edges[:-1], "value": m.model.prof.p}))
            for p, m in lr_models.items():
                for end, prof in [("5p", m.end5), ("3p", m.end3)]:
                    bias.append(pd.DataFrame({"fit": fit, "model": f"LR truncation {end} {p}", "x": prof.edges[:-1], "value": prof.p}))
            for p, m in (lr_length or {}).items():
                grid = np.array([300, 500, 1000, 2000, 4000, 8000, 16000])
                bias.append(pd.DataFrame({"fit": fit, "model": f"LR length {p}", "x": grid, "value": m.rates_at(grid)}))
        summary.append({"fit": fit, "n_libraries": len(libs), "units": len(present), "loglik": res.loglik[-1],
                        "n_iter": res.n_iter, "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)

    spliced = np.array([not n.endswith("-U") for n in names])
    sp_names = np.array(names)[spliced]
    rows = []
    for u in both:
        for a, b in [("SR_plain", "LR_plain"), ("SR", "LR")]:
            ka, kb = f"{a}|{u}", f"{b}|{u}"
            if ka in tpm and kb in tpm:
                rows.append({"cell_type": u, "a": a, "b": b,
                             **concordance_metrics(pd.Series(tpm[ka][spliced], index=sp_names),
                                                   pd.Series(tpm[kb][spliced], index=sp_names), t2g, min_gene_tpm=20)})
    conc = pd.DataFrame(rows)
    if len(conc):
        print(conc.groupby(["a", "b"])[["pearson_log1p", "psi_pearson", "gene_tv_median", "major_isoform_agree"]]
              .mean().round(3).to_string())
    conc.to_csv(out_dir / "concordance.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "fits.tsv", sep="\t", index=False)
    if bias:
        pd.concat(bias).to_csv(out_dir / "bias.tsv", sep="\t", index=False)
    pd.DataFrame(tpm, index=names).to_csv(out_dir / "tpm.tsv.gz", sep="\t", float_format="%.6g")


if __name__ == "__main__":
    main(*sys.argv[1:8])
