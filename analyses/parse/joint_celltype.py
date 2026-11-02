"""
Joint Parse short-read + single-cell long-read quantification by cell type.

Units are CellTypist cell types (Immune_All_Low, the vocabulary of both the
short-read recipes and the long-read labels). Libraries:
  SR  one per (series, subject, cell type, primer): Parse short-read pseudobulks
      (salmon/alevin-fry RAD positions)
  LR  one per (cell type, kit, source library) from each long-read directory:
      Kinnex Parse (kit = RT primer) and/or Kinnex 10x 3'/5' (kit = poly(dT)@3p / @5p)
Base rates (alevin.primer_rates): poly(dT) 1 on spliced targets and kappa0 (L - r + 1) on
unspliced targets (internal priming; kappa0 = e^-8.6, the scale learned from Parse poly(dT)
against hexamer), random hexamer L - r + 1 (r = 150 for SR, 0 for LR). The learned unspliced
scales are corrections to kappa0 with a weak ridge.
Each library belongs to the unit of its cell type. Short- and long-read donors
differ, so a unit's abundances are shared across donors and platforms: short
reads bring coverage, long reads bring isoform identifiability.

Bias models are shared per platform x primer/kit across all of its libraries:
  SR position profile (poly(dT): distance to 3' end; hexamer: relative position)
  LR read ends (AnchoredTruncationModel: anchored end profile and censored read length,
  3'/5'-anchored mixture), per kit
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
from joli.lr_bias import (AnchoredTruncationModel, FeatureColumns, LengthBias, ReadPositionModel,
                          StackedFeatures, fit_bias)
from joli.paf import read_fasta_lengths
from joli.rad import libraries_from_molecules, load_rad_dump, molecule_alignments

SR_READ_LEN = 150
# final fits: high step cap, and stop once the log posterior gains < 1e-10 relative per iterate
RUN_KW = dict(max_iter=20000, ll_rel_tol=1e-10)
KAPPA0 = np.exp(-8.6)       # poly(dT) internal priming per base (Parse dT + hexamer joint fits)
PRIMERS = ("poly(dT)", "random hexamer")


def sr_sources(series_dirs):
    """Short-read RAD dumps: list of (series dir, library, group label per local cell index)."""
    out = []
    for sdir in series_dirs:
        sdir = Path(sdir)
        cells = pd.read_csv(sdir / "cells.tsv", sep="\t")
        for lib in pd.read_csv(sdir / "libraries.tsv", sep="\t")["library"]:
            c = cells[cells["library"] == lib].reset_index(drop=True)
            groups = np.array([f"{t}|SR|{p}|{sdir.name}:{s}" for t, p, s in zip(c["cell_type"], c["primer"], c["subject"])],
                              dtype=object)
            out.append((sdir, lib, c["barcode"].to_numpy(), groups))
    return out


def lr_sources(kdirs):
    """
    Long-read sources: list of (files, group label per cell-half index, cell type per cell) per
    source library. Sources are molecules/alignments.npz, or molecules/chunk_<library>_*.npz when
    reads are already one per molecule (one source per library); cells index molecules/cells.txt.

    Each (cell, primer half) gets a group label "<cell type>|LR|<kit>|<dir>:<library>".
    The kit names the read-end/length behaviour: the Parse RT primer for Kinnex
    Parse (two halves per cell), or "poly(dT)@3p" / "poly(dT)@5p" for 10x 3'/5'
    (cells.tsv has kit and library columns; one half per cell).
    """
    out = []
    for kdir in kdirs:
        kdir = Path(kdir)
        cell_names = (kdir / "molecules" / "cells.txt").read_text().split()
        ct = pd.read_csv(kdir / "celltypist.tsv", sep="\t").set_index("cell")["cell_type"]
        types = ct.reindex(cell_names).to_numpy(dtype=object)
        info = pd.read_csv(kdir / "cells.tsv", sep="\t").drop_duplicates("cell").set_index("cell").reindex(cell_names)
        groups = []
        for t, (_, r) in zip(types, info.iterrows()):
            if not isinstance(t, str):
                groups += [None, None]
            elif "kit" in info.columns:
                groups += [f"{t}|LR|{r['primer']}@{r['kit']}|{kdir.name}:{r['library']}", None]
            else:
                groups += [f"{t}|LR|{p}|{kdir.name}:parse" for p in PRIMERS]
        groups = np.array(groups, dtype=object)
        if (kdir / "molecules" / "alignments.npz").exists():
            out.append(([kdir / "molecules" / "alignments.npz"], groups, types))
        else:   # streamed: per-chunk alignment files named chunk_<library>_<k>.npz
            files = sorted((kdir / "molecules").glob("chunk_*.npz"))
            by_lib = {}
            for f in files:
                by_lib.setdefault(f.stem.rsplit("_", 1)[0], []).append(f)
            out += [(fs, groups, types) for fs in by_lib.values()]
    return out


def sr_libraries(sources, names, fraction, rates, model):
    """Libraries of the groups in `rates`, built one RAD dump at a time."""
    libs = {}
    for sdir, lib, barcodes, groups in sources:
        if not any(g in rates for g in set(groups)):
            continue
        t0 = time.time()
        dump = load_rad_dump(sdir / lib / "rad_dump")
        if dump["refs"] != names:
            raise ValueError(f"{sdir.name}/{lib}: RAD reference order differs from spliceu.fa")
        a = molecule_alignments(dump, pd.Series(np.arange(len(barcodes)), index=barcodes), fraction=fraction)
        del dump
        a = a[pd.Series(groups[a["cell"].to_numpy()]).isin(rates).to_numpy()]
        new = libraries_from_molecules(a, groups, names, rates, model=model)
        libs.update(new)
        print(f"SR {sdir.name}/{lib}: {a['read'].nunique()} molecules -> {len(new)} libraries, "
              f"{sum(len(l.tcc.ec_counts) for l in new.values())} ECs ({time.time() - t0:.0f} s)", flush=True)
    return libs


def lr_libraries(sources, names, fraction, rates, model, seed=0):
    """Libraries of the groups in `rates`, built one source library at a time (random molecule fraction)."""
    libs = {}
    rng = np.random.default_rng(seed)
    for files, groups, _ in sources:
        t0 = time.time()
        parts, off = [], 0
        for f in files:
            z = np.load(f)
            mol = z["mol"] if "mol" in z else pd.factorize(z["read"])[0]
            keep = rng.random(int(mol.max()) + 1) < fraction
            cell = 2 * z["cell"].astype(np.int64) + z["half"]
            sel = keep[mol] & pd.Series(groups[cell]).isin(rates).to_numpy()
            parts.append(pd.DataFrame({"read": mol[sel].astype(np.int64) + off,
                                       "cell": cell[sel], "tx_idx": z["tx_idx"][sel], "start": z["tstart"][sel],
                                       "end": z["tend"][sel], "tlen": z["tlen"][sel]}))
            off += int(mol.max()) + 1
        a = pd.concat(parts, ignore_index=True)
        del parts
        new = libraries_from_molecules(a, groups, names, rates, model=model, platform="long")
        libs.update(new)
        print(f"LR {files[0].stem.rsplit('_', 1)[0]} ({len(files)} files): {a['read'].nunique()} molecules -> "
              f"{len(new)} libraries, {sum(len(l.tcc.ec_counts) for l in new.values())} ECs ({time.time() - t0:.0f} s)",
              flush=True)
    return libs


def main(sr_dirs, kdir, ref_dir, out_dir, sr_fraction=0.1, lr_fraction=1.0, min_cells=100):
    ref_dir, out_dir = Path(ref_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, tx_len = read_fasta_lengths(ref_dir / "spliceu.fa")
    unspliced = np.array([n.endswith("-U") for n in names])
    sr_src = sr_sources(sr_dirs.split(","))
    lr_src = lr_sources(kdir.split(","))

    # units: cell types with >= min_cells in either platform (two primer barcodes per SR cell)
    sr_cells = pd.Series([g.split("|")[0] for _, _, _, gs in sr_src for g in gs]).value_counts() // 2
    lr_cells = pd.Series([t for _, _, ts in {id(s[2]): s for s in lr_src}.values() for t in ts
                          if isinstance(t, str)]).value_counts()
    n_cells = sr_cells.add(lr_cells, fill_value=0)
    units = sorted(n_cells[(sr_cells.reindex(n_cells.index, fill_value=0) >= int(min_cells))
                           | (lr_cells.reindex(n_cells.index, fill_value=0) >= int(min_cells))].index)
    both = [u for u in units if sr_cells.get(u, 0) >= int(min_cells) and lr_cells.get(u, 0) >= int(min_cells)]
    print(f"{len(units)} units, {len(both)} with both platforms: {both}", flush=True)
    unit_idx = {u: k for k, u in enumerate(units)}

    sr_feat = StackedFeatures([ReadPositionModel(tx_len, SR_READ_LEN, "3p", uniform=unspliced),
                               ReadPositionModel(tx_len, SR_READ_LEN, "relative", uniform=unspliced)])
    def rate(g):
        plat, kit = g.split("|")[1:3]
        return primer_rates(kit.split("@")[0], tx_len, SR_READ_LEN if plat == "SR" else 0, unspliced, kappa=KAPPA0)
    keep_g = lambda gs: {g: rate(g) for g in set(x for x in gs if x is not None) if g.split("|")[0] in unit_idx}
    t0 = time.time()
    sr_libs = sr_libraries(sr_src, names, float(sr_fraction), keep_g([g for s in sr_src for g in s[3]]), sr_feat)
    lr_libs = lr_libraries(lr_src, names, float(lr_fraction), keep_g({g for s in lr_src for g in s[1]}),
                           AnchoredTruncationModel(tx_len, uniform=unspliced))
    n_pos = sum(len(l.tcc.ec_transcripts.values) for l in {**sr_libs, **lr_libs}.values())
    print(f"{len(sr_libs)} SR and {len(lr_libs)} LR libraries, {n_pos} EC positions ({time.time() - t0:.0f} s)", flush=True)

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
                    trunc[i] = lr_models.setdefault(primer, AnchoredTruncationModel(tx_len, uniform=unspliced))
                    if lr_length is not None:
                        length[i] = lr_length.setdefault(primer, LengthBias(tx_len, extra=unspliced[:, None], ridge_extra=1e-3))
        return keys, libs, unit_of, trunc, length

    # LR alone (with truncation) is the baseline; SR alone cannot identify isoforms, so it is not fitted
    fits = {"LR": (lr_libs, True, False),
            "joint_plain": ({**sr_libs, **lr_libs}, False, False),
            "joint": ({**sr_libs, **lr_libs}, True, True)}
    tpm, summary, bias, done = {}, [], [], {}
    for fit, (lib_dict, use_models, use_length) in fits.items():
        t0 = time.time()
        sr_models = {"poly(dT)": FeatureColumns(ReadPositionModel(tx_len, SR_READ_LEN, "3p", uniform=unspliced), [0]),
                     "random hexamer": FeatureColumns(ReadPositionModel(tx_len, SR_READ_LEN, "relative", uniform=unspliced), [1])}
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
        # the full joint fit starts from the joint fit without bias models (same libraries and units)
        init = done.get("joint_plain") if fit == "joint" else None
        init = np.maximum(init[1], 1e-10) if init is not None and init[0] == present else None
        if trunc or length:
            res = fit_bias(em, trunc=trunc, length=length, n_outer=8, verbose=True, run_kw=RUN_KW, init_theta=init)
        else:
            res = em.run(init_theta=init, **RUN_KW)
        done[fit] = (present, res.theta)
        for k in present:
            tpm[f"{fit}|{units[k]}"] = molar_to_tpm(res.theta[local[k]])
        if use_models:
            for p, m in sr_models.items():
                bias.append(pd.DataFrame({"fit": fit, "model": f"SR position {p}", "x": m.model.prof.edges[:-1], "value": m.model.prof.p}))
            for p, m in lr_models.items():
                q = m.profiles()
                bias.append(pd.DataFrame({"fit": fit, "model": f"LR {p} " + q["anchor"] + "-anchored " + q["kind"],
                                          "x": q["x"], "value": q["value"]}))
                bias.append(pd.DataFrame({"fit": fit, "model": f"LR {p} unspliced read_length",
                                          "x": m.length.edges[:-1], "value": m.length.p}))
            for p, m in (lr_length or {}).items():
                grid = np.array([300, 500, 1000, 2000, 4000, 8000, 16000])
                bias.append(pd.DataFrame({"fit": fit, "model": f"LR length {p}", "x": grid, "value": m.rates_at(grid)}))
                bias.append(pd.DataFrame({"fit": fit, "model": f"LR log unspliced scale {p}", "x": [0.0], "value": [m.w[-1]]}))
        summary.append({"fit": fit, "n_libraries": len(libs), "units": len(present), "loglik": res.loglik[-1],
                        "sr_log_kappa_unspliced": float(sr_kappa.w[0]) if sr_kappa is not None else np.nan,
                        "n_iter": res.n_iter, "converged": res.converged, "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)

    pd.DataFrame(summary).to_csv(out_dir / "fits.tsv", sep="\t", index=False)
    if bias:
        pd.concat(bias).to_csv(out_dir / "bias.tsv", sep="\t", index=False)
    pd.DataFrame(tpm, index=names).to_csv(out_dir / "tpm.tsv.gz", sep="\t", float_format="%.6g")


if __name__ == "__main__":
    main(*sys.argv[1:8])
