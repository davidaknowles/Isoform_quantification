"""
Joint Parse short-read + single-cell long-read quantification by cell type.

Units are CellTypist cell types (Immune_All_Low, the vocabulary of both the
short-read recipes and the long-read labels). Libraries:
  SR  one per (series, subject, cell type, primer): Parse short-read pseudobulks
      (salmon/alevin-fry RAD positions)
  LR  one per (cell type, kit, source library) from each long-read directory:
      Kinnex Parse (kit = RT primer) and/or Kinnex 10x 3'/5' (kit = poly(dT)@3p / @5p)
Base rates (alevin.primer_rates): poly(dT) 1 on spliced targets and L - r + 1 on
unspliced targets (internal priming), random hexamer L - r + 1 (r = 150 for SR, 0 for LR).
Each library belongs to the unit of its cell type. Short- and long-read donors
differ, so a unit's abundances are shared across donors and platforms: short
reads bring coverage, long reads bring isoform identifiability.

Bias models are shared per platform x primer/kit across all of its libraries:
  SR position profile (poly(dT): distance to 3' end; hexamer: relative position)
  LR read-end truncation (5' and 3' uncovered bases), per kit
  LR length bias with a separate unspliced scale, per kit (identifiable against
  SR, whose spliced rates are fixed)
  SR poly(dT) scale of internal priming on unspliced targets

Fits: LR (long reads alone, with truncation; baseline), joint_plain, joint (all bias models).
Outputs: tpm.tsv.gz (transcripts x fit|unit), fits.tsv, bias.tsv.

Run: python analyses/parse/joint_celltype.py <sr_series_dirs,comma-separated> <lr_dirs,comma-separated> <ref_dir> <out_dir>
                                             [sr_fraction] [lr_fraction] [min_cells]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.alevin import primer_rates
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


def load_lr(kdirs, fraction, seed=0):
    """
    Long-read molecules of one or more processed long-read directories
    (molecules/alignments.npz, molecules/cells.txt, celltypist.tsv, cells.tsv).

    Each (cell, primer half) gets a group label "<cell type>|LR|<kit>|<dir>:<library>".
    The kit names the read-end/length behaviour: the Parse RT primer for Kinnex
    Parse (two halves per cell), or "poly(dT)@3p" / "poly(dT)@5p" for 10x 3'/5'
    (cells.tsv has kit and library columns; one half per cell).
    Returns (alignments with global cell index, groups per cell-half, cell type per cell).
    """
    parts, groups, types_all, mol_off, cell_off = [], [], [], 0, 0
    for kdir in kdirs:
        kdir = Path(kdir)
        z = np.load(kdir / "molecules" / "alignments.npz")
        mol = z["mol"]
        keep = np.random.default_rng(seed).random(int(mol.max()) + 1) < fraction
        sel = keep[mol]
        aln = pd.DataFrame({"read": mol[sel] + mol_off, "cell": cell_off + 2 * z["cell"][sel].astype(np.int64) + z["half"][sel],
                            "tx_idx": z["tx_idx"][sel], "start": z["tstart"][sel], "end": z["tend"][sel],
                            "tlen": z["tlen"][sel]})
        mol_off += int(mol.max()) + 1
        cell_names = (kdir / "molecules" / "cells.txt").read_text().split()
        ct = pd.read_csv(kdir / "celltypist.tsv", sep="\t").set_index("cell")["cell_type"]
        types = ct.reindex(cell_names).to_numpy(dtype=object)
        info = pd.read_csv(kdir / "cells.tsv", sep="\t").drop_duplicates("cell").set_index("cell").reindex(cell_names)
        for t, (_, r) in zip(types, info.iterrows()):
            if not isinstance(t, str):
                groups += [None, None]
            elif "kit" in info.columns:
                groups += [f"{t}|LR|{r['primer']}@{r['kit']}|{kdir.name}:{r['library']}", None]
            else:
                groups += [f"{t}|LR|{p}|{kdir.name}:parse" for p in PRIMERS]
        types_all += list(types)
        cell_off += 2 * len(cell_names)
        parts.append(aln)
    groups = np.array(groups, dtype=object)
    aln = pd.concat(parts, ignore_index=True)
    aln = aln[pd.notna(groups[aln["cell"].to_numpy()])]
    print(f"LR: {aln['read'].nunique()} molecules in labelled cells", flush=True)
    return aln, groups, np.array(types_all, dtype=object)


def main(sr_dirs, kdir, ref_dir, out_dir, sr_fraction=0.1, lr_fraction=1.0, min_cells=100):
    ref_dir, out_dir = Path(ref_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, tx_len = read_fasta_lengths(ref_dir / "spliceu.fa")
    unspliced = np.array([n.endswith("-U") for n in names])

    sr_aln, sr_groups = load_sr(sr_dirs.split(","), names, float(sr_fraction))
    lr_aln, lr_groups, lr_types = load_lr(kdir.split(","), float(lr_fraction))

    # units: cell types with >= min_cells in either platform
    sr_cells = pd.Series([g.split("|")[0] for g in sr_groups]).value_counts() // 2
    lr_cells = pd.Series([t for t in lr_types if isinstance(t, str)]).value_counts()
    n_cells = sr_cells.add(lr_cells, fill_value=0)
    units = sorted(n_cells[(sr_cells.reindex(n_cells.index, fill_value=0) >= int(min_cells))
                           | (lr_cells.reindex(n_cells.index, fill_value=0) >= int(min_cells))].index)
    both = [u for u in units if sr_cells.get(u, 0) >= int(min_cells) and lr_cells.get(u, 0) >= int(min_cells)]
    print(f"{len(units)} units, {len(both)} with both platforms: {both}", flush=True)
    unit_idx = {u: k for k, u in enumerate(units)}

    sr_feat = StackedFeatures([ReadPositionModel(tx_len, SR_READ_LEN, "3p"),
                               ReadPositionModel(tx_len, SR_READ_LEN, "relative")])
    def rate(g):
        plat, kit = g.split("|")[1:3]
        return primer_rates(kit.split("@")[0], tx_len, SR_READ_LEN if plat == "SR" else 0, unspliced)
    keep_g = lambda gs: {g: rate(g) for g in set(x for x in gs if x is not None) if g.split("|")[0] in unit_idx}
    sr_aln = sr_aln[pd.Series(sr_groups[sr_aln["cell"].to_numpy()]).isin(keep_g(sr_groups)).to_numpy()]
    lr_aln = lr_aln[pd.Series(lr_groups[lr_aln["cell"].to_numpy()]).isin(keep_g(lr_groups)).to_numpy()]
    sr_libs = libraries_from_molecules(sr_aln, sr_groups, names, keep_g(sr_groups), model=sr_feat)
    lr_libs = libraries_from_molecules(lr_aln, lr_groups, names, keep_g(lr_groups),
                                       model=TruncationModel(tx_len), platform="long")
    del sr_aln, lr_aln
    print(f"{len(sr_libs)} SR and {len(lr_libs)} LR libraries", flush=True)

    def setup(lib_dict, use_models, sr_models, lr_models, lr_length, sr_kappa):
        keys = sorted(lib_dict)
        libs = [lib_dict[k] for k in keys]
        unit_of = [unit_idx[k.split("|")[0]] for k in keys]
        trunc, length = {}, {}
        if use_models:
            for i, k in enumerate(keys):
                plat, primer = k.split("|")[1:3]
                if plat == "SR":
                    trunc[i] = sr_models[primer]
                    if sr_kappa is not None and primer == "poly(dT)":
                        length[i] = sr_kappa
                else:
                    # one truncation and one length model per long-read kit, created on first use
                    trunc[i] = lr_models.setdefault(primer, TruncationModel(tx_len))
                    if lr_length is not None:
                        length[i] = lr_length.setdefault(primer, LengthBias(tx_len, extra=unspliced[:, None]))
        return keys, libs, unit_of, trunc, length

    # LR alone (with truncation) is the baseline; SR alone cannot identify isoforms, so it is not fitted
    fits = {"LR": (lr_libs, True, False),
            "joint_plain": ({**sr_libs, **lr_libs}, False, False),
            "joint": ({**sr_libs, **lr_libs}, True, True)}
    tpm, summary, bias = {}, [], []
    for fit, (lib_dict, use_models, use_length) in fits.items():
        t0 = time.time()
        sr_models = {"poly(dT)": FeatureColumns(ReadPositionModel(tx_len, SR_READ_LEN, "3p"), [0]),
                     "random hexamer": FeatureColumns(ReadPositionModel(tx_len, SR_READ_LEN, "relative"), [1])}
        lr_models = {}
        # LR: smooth length bias plus a separate scale for unspliced targets; SR poly(dT): the
        # scale of internal priming on unspliced targets (SR hexamer and SR poly(dT) spliced rates fixed)
        lr_length = {} if use_length else None
        sr_kappa = LengthBias(tx_len, extra=unspliced[:, None], length_terms=False, ridge=1e-3) if use_length else None
        keys, libs, unit_of, trunc, length = setup(lib_dict, use_models, sr_models, lr_models, lr_length, sr_kappa)
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
                        "sr_log_kappa_unspliced": float(sr_kappa.w[0]) if sr_kappa is not None else np.nan,
                        "n_iter": res.n_iter, "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)

    pd.DataFrame(summary).to_csv(out_dir / "fits.tsv", sep="\t", index=False)
    if bias:
        pd.concat(bias).to_csv(out_dir / "bias.tsv", sep="\t", index=False)
    pd.DataFrame(tpm, index=names).to_csv(out_dir / "tpm.tsv.gz", sep="\t", float_format="%.6g")


if __name__ == "__main__":
    main(*sys.argv[1:8])
