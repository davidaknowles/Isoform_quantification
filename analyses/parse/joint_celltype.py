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


def lr_libraries(sources, names, fraction, rates, model, seed=0, test_fraction=0.0):
    """
    Libraries of the groups in `rates`, built one source library at a time. Each molecule draws
    u ~ U(0, 1) (fixed seed): u < test_fraction goes to held-out libraries, and training keeps
    test_fraction <= u < test_fraction + fraction (1 - test_fraction), so training sets of
    different `fraction` are nested and share one held-out set. Returns the training libraries,
    or (training, held-out) when test_fraction > 0.
    """
    libs, test_libs = {}, {}
    rng = np.random.default_rng(seed)
    lo, hi = test_fraction, test_fraction + fraction * (1.0 - test_fraction)
    for files, groups, _ in sources:
        t0 = time.time()
        parts, tparts, off = [], [], 0
        for f in files:
            z = np.load(f)
            mol = z["mol"] if "mol" in z else pd.factorize(z["read"])[0]
            u = rng.random(int(mol.max()) + 1)[mol]
            cell = 2 * z["cell"].astype(np.int64) + z["half"]
            ok = pd.Series(groups[cell]).isin(rates).to_numpy()
            for dest, sel in ((parts, ok & (u >= lo) & (u < hi)), (tparts, ok & (u < lo))):
                if sel.any():
                    dest.append(pd.DataFrame({"read": mol[sel].astype(np.int64) + off,
                                              "cell": cell[sel], "tx_idx": z["tx_idx"][sel], "start": z["tstart"][sel],
                                              "end": z["tend"][sel], "tlen": z["tlen"][sel]}))
            off += int(mol.max()) + 1
        a = pd.concat(parts, ignore_index=True)
        del parts
        new = libraries_from_molecules(a, groups, names, rates, model=model, platform="long")
        libs.update(new)
        if tparts:
            test_libs.update(libraries_from_molecules(pd.concat(tparts, ignore_index=True), groups, names, rates,
                                                      model=model, platform="long"))
        del tparts
        print(f"LR {files[0].stem.rsplit('_', 1)[0]} ({len(files)} files): {a['read'].nunique()} molecules -> "
              f"{len(new)} libraries, {sum(len(l.tcc.ec_counts) for l in new.values())} ECs ({time.time() - t0:.0f} s)",
              flush=True)
    return (libs, test_libs) if test_fraction > 0 else libs


def prepare(sr_dirs, kdir, ref_dir, min_cells=100):
    """Reference, sources, units (cell types with >= min_cells in either platform) and base rates."""
    ref_dir = Path(ref_dir)
    names, tx_len = read_fasta_lengths(ref_dir / "spliceu.fa")
    unspliced = np.array([n.endswith("-U") for n in names])
    sr_src = sr_sources(sr_dirs.split(","))
    lr_src = lr_sources(kdir.split(","))
    # two primer barcodes per SR cell
    sr_cells = pd.Series([g.split("|")[0] for _, _, _, gs in sr_src for g in gs]).value_counts() // 2
    lr_cells = pd.Series([t for _, _, ts in {id(s[2]): s for s in lr_src}.values() for t in ts
                          if isinstance(t, str)]).value_counts()
    n_cells = sr_cells.add(lr_cells, fill_value=0)
    units = sorted(n_cells[(sr_cells.reindex(n_cells.index, fill_value=0) >= int(min_cells))
                           | (lr_cells.reindex(n_cells.index, fill_value=0) >= int(min_cells))].index)
    both = [u for u in units if sr_cells.get(u, 0) >= int(min_cells) and lr_cells.get(u, 0) >= int(min_cells)]
    print(f"{len(units)} units, {len(both)} with both platforms: {both}", flush=True)
    unit_idx = {u: k for k, u in enumerate(units)}

    def rate(g):
        plat, kit = g.split("|")[1:3]
        return primer_rates(kit.split("@")[0], tx_len, SR_READ_LEN if plat == "SR" else 0, unspliced, kappa=KAPPA0)
    keep_g = lambda gs: {g: rate(g) for g in set(x for x in gs if x is not None) if g.split("|")[0] in unit_idx}
    # gene groups for platform-specific gene totals: unspliced targets form groups of their own
    t2g = pd.read_csv(ref_dir / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    gcode = pd.factorize(t2g.reindex(names))[0]
    n_genes = int(gcode.max()) + 1
    gene_groups = np.where(gcode < 0, -1, gcode + np.where(unspliced, n_genes, 0))
    return dict(names=names, tx_len=tx_len, unspliced=unspliced, sr_src=sr_src, lr_src=lr_src,
                units=units, unit_idx=unit_idx, keep_g=keep_g, genes=gcode, gene_groups=gene_groups)


def build_libraries(ctx, sr_fraction, lr_fraction, lr_test_fraction=0.0):
    """SR and LR libraries (and held-out LR libraries when lr_test_fraction > 0; else None)."""
    names, tx_len, unspliced = ctx["names"], ctx["tx_len"], ctx["unspliced"]
    sr_feat = StackedFeatures([ReadPositionModel(tx_len, SR_READ_LEN, "3p", uniform=unspliced),
                               ReadPositionModel(tx_len, SR_READ_LEN, "relative", uniform=unspliced)])
    t0 = time.time()
    sr_libs = {} if float(sr_fraction) <= 0 else sr_libraries(
        ctx["sr_src"], names, float(sr_fraction), ctx["keep_g"]([g for s in ctx["sr_src"] for g in s[3]]), sr_feat)
    lr = lr_libraries(ctx["lr_src"], names, float(lr_fraction), ctx["keep_g"]({g for s in ctx["lr_src"] for g in s[1]}),
                      AnchoredTruncationModel(tx_len, uniform=unspliced), test_fraction=float(lr_test_fraction))
    lr_libs, lr_test = lr if float(lr_test_fraction) > 0 else (lr, None)
    n_pos = sum(len(l.tcc.ec_transcripts.values) for l in {**sr_libs, **lr_libs}.values())
    print(f"{len(sr_libs)} SR and {len(lr_libs)} LR libraries, {n_pos} EC positions ({time.time() - t0:.0f} s)", flush=True)
    return sr_libs, lr_libs, lr_test


def fit_one(ctx, lib_dict, use_models, use_length, init=None, verbose=True, platform_genes=False, pool=None,
            pool_tau=1.0):
    """
    Fit one configuration. Bias models are shared per platform x primer/kit: SR position profiles
    (poly(dT): distance to the 3' end; hexamer: relative position) and the SR poly(dT) unspliced
    scale; per LR kit an AnchoredTruncationModel and a LengthBias with an unspliced scale.
    Units are cell types, or with platform_genes (cell type, platform) pairs whose abundances
    share within-gene fractions but have platform-specific gene totals (JointEM tie/groups;
    unspliced targets are groups of their own). pool ("gene" | "transcript") instead shrinks the
    platforms' gene totals (with shared within-gene fractions) or transcript abundances towards a
    shared cell-type profile with log-scale sd pool_tau (JointEM pool).
    init: (unit labels, theta) of an earlier fit; per-cell-type abundances are copied to both
    platforms of a platform_genes fit.
    Returns dict with res, present (unit labels), label (library key -> unit label), keys and models.
    """
    tx_len, unspliced, unit_idx = ctx["tx_len"], ctx["unspliced"], ctx["unit_idx"]
    sr_models = {"poly(dT)": FeatureColumns(ReadPositionModel(tx_len, SR_READ_LEN, "3p", uniform=unspliced), [0]),
                 "random hexamer": FeatureColumns(ReadPositionModel(tx_len, SR_READ_LEN, "relative", uniform=unspliced), [1])}
    lr_models = {}
    lr_length = {} if use_length else None
    sr_kappa = LengthBias(tx_len, extra=unspliced[:, None], length_terms=False, ridge=1e-3) if use_length else None
    keys = sorted(lib_dict)
    libs = [lib_dict[k] for k in keys]
    platform_genes = platform_genes or pool is not None
    label = (lambda k: (k.split("|")[0], k.split("|")[1])) if platform_genes else (lambda k: k.split("|")[0])
    trunc, length = {}, {}
    if use_models:
        for i, k in enumerate(keys):
            plat, primer = k.split("|")[1:3]
            if plat == "SR":
                trunc[i] = sr_models[primer]
                if sr_kappa is not None and primer == "poly(dT)":
                    length[i] = sr_kappa
            else:
                trunc[i] = lr_models.setdefault(primer, AnchoredTruncationModel(tx_len, uniform=unspliced))
                if lr_length is not None:
                    length[i] = lr_length.setdefault(primer, LengthBias(tx_len, extra=unspliced[:, None], ridge_extra=1e-3))
    present = sorted({label(k) for k in keys})
    local = {u: i for i, u in enumerate(present)}
    pi = np.zeros((len(libs), len(present)))
    pi[np.arange(len(libs)), [local[label(k)] for k in keys]] = 1.0
    tie = [unit_idx[u[0]] for u in present] if platform_genes else None
    em = JointEM(libs, pi=pi, unit_names=["@".join(u) if platform_genes else u for u in present],
                 tie=tie, groups=ctx["gene_groups"] if platform_genes else None, pool=pool, pool_tau=pool_tau)
    init_theta = None
    if init is not None:
        imap = {u: i for i, u in enumerate(init[0])}
        rows = [imap.get(u, imap.get(u[0]) if isinstance(u, tuple) else None) for u in present]
        if all(r is not None for r in rows):
            init_theta = np.maximum(init[1][rows], 1e-10)
    if trunc or length:
        res = fit_bias(em, trunc=trunc, length=length, n_outer=8, verbose=verbose, run_kw=RUN_KW, init_theta=init_theta)
    else:
        res = em.run(init_theta=init_theta, **RUN_KW)
    return dict(res=res, present=present, label=label, keys=keys, sr_models=sr_models, lr_models=lr_models,
                lr_length=lr_length, sr_kappa=sr_kappa)


def main(sr_dirs, kdir, ref_dir, out_dir, sr_fraction=0.1, lr_fraction=1.0, min_cells=100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = prepare(sr_dirs, kdir, ref_dir, min_cells)
    sr_libs, lr_libs, _ = build_libraries(ctx, sr_fraction, lr_fraction)
    units = ctx["units"]

    # LR alone (with truncation) is the baseline; SR alone cannot identify isoforms, so it is not fitted
    fits = {"LR": (lr_libs, True, False),
            "joint_plain": ({**sr_libs, **lr_libs}, False, False),
            "joint": ({**sr_libs, **lr_libs}, True, True)}
    tpm, summary, bias, done = {}, [], [], {}
    for fit, (lib_dict, use_models, use_length) in fits.items():
        t0 = time.time()
        # the full joint fit starts from the joint fit without bias models (same libraries and units)
        out = fit_one(ctx, lib_dict, use_models, use_length, init=done.get("joint_plain") if fit == "joint" else None)
        res, present = out["res"], out["present"]
        done[fit] = (present, res.theta)
        for i, u in enumerate(present):
            tpm[f"{fit}|{'@'.join(u) if isinstance(u, tuple) else u}"] = molar_to_tpm(res.theta[i])
        if use_models:
            bias.append(bias_table(fit, out))
        sr_kappa = out["sr_kappa"]
        summary.append({"fit": fit, "n_libraries": len(out["keys"]), "units": len(present), "loglik": res.loglik[-1],
                        "sr_log_kappa_unspliced": float(sr_kappa.w[0]) if sr_kappa is not None else np.nan,
                        "n_iter": res.n_iter, "converged": res.converged, "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)

    pd.DataFrame(summary).to_csv(out_dir / "fits.tsv", sep="\t", index=False)
    if bias:
        pd.concat(bias).to_csv(out_dir / "bias.tsv", sep="\t", index=False)
    pd.DataFrame(tpm, index=ctx["names"]).to_csv(out_dir / "tpm.tsv.gz", sep="\t", float_format="%.6g")


def bias_table(fit, out):
    """Long-format table of a fit's bias models."""
    rows = []
    for p, m in out["sr_models"].items():
        rows.append(pd.DataFrame({"fit": fit, "model": f"SR position {p}", "x": m.model.prof.edges[:-1], "value": m.model.prof.p}))
    for p, m in out["lr_models"].items():
        q = m.profiles()
        rows.append(pd.DataFrame({"fit": fit, "model": f"LR {p} " + q["anchor"] + "-anchored " + q["kind"],
                                  "x": q["x"], "value": q["value"]}))
        rows.append(pd.DataFrame({"fit": fit, "model": f"LR {p} unspliced read_length",
                                  "x": m.length.edges[:-1], "value": m.length.p}))
    for p, m in (out["lr_length"] or {}).items():
        grid = np.array([300, 500, 1000, 2000, 4000, 8000, 16000])
        rows.append(pd.DataFrame({"fit": fit, "model": f"LR length {p}", "x": grid, "value": m.rates_at(grid)}))
        rows.append(pd.DataFrame({"fit": fit, "model": f"LR log unspliced scale {p}", "x": [0.0], "value": [m.w[-1]]}))
    return pd.concat(rows)


if __name__ == "__main__":
    main(*sys.argv[1:8])
