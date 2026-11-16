"""
Mouse-brain pilot fits: Tabula Muris Senis Smart-seq2 pseudobulks (region x cell class x age) and ENCODE
bulk PacBio (cortex, hippocampus), from build_libraries.py.

Fits:
  SR_only  each SR group its own unit (FragmentModel per library): Smart-seq2 alone
  LR_bulk  one unit per bulk tissue (AnchoredTruncationModel shared by the LR libraries): bulk LR alone
  joint    base units (region, class) carry transcript abundances with a design prior (intercept + region
           + lineage + class effects, unit noise TAU); each SR group is its base unit rescaled per gene
           (design_gene) with one per-gene SR scale profile shared by all SR units (scale_groups), so SR
           gene expression informs the base units; each bulk LR library is a mixture over its region's base
           units with composition estimated (fit_bias composition). Base units of regions without bulk LR
           (striatum, cerebellum) are informed by SR and the prior.
Outputs: tpm.tsv.gz (transcripts x fit|unit; joint base units "<region>|<class>", SR units
"<region>|<class>|<age>@SR", consensus profiles "<region>|<class>|consensus"), composition.tsv, fits.tsv.

Run: python analyses/mouse_brain/fit.py <libraries.pkl> <out_dir> [ct_sd]
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import AnchoredTruncationModel, FragmentModel, fit_bias

LINEAGE = {"neuron": "neuron", "astrocyte": "glia", "oligodendrocyte": "glia", "OPC": "glia", "microglia": "immune",
           "endothelial": "vascular", "pericyte": "vascular"}
TAU = 1.0
REGION_SD, LINEAGE_SD = 2.0, 2.0
RUN_KW = dict(max_iter=20000, ll_rel_tol=1e-10)


def design(base_units, ct_sd):
    """Design rows of (region, class) base units: intercept, region, lineage and class indicators."""
    regions = sorted({r for r, _ in base_units})
    lins = sorted(set(LINEAGE.values()))
    classes = sorted({c for _, c in base_units})
    cols = ["intercept"] + [f"region:{r}" for r in regions] + [f"lineage:{l}" for l in lins] + [f"class:{c}" for c in classes]
    sd = np.r_[np.inf, np.full(len(regions), REGION_SD), np.full(len(lins), LINEAGE_SD), np.full(len(classes), ct_sd)]
    X = np.zeros((len(base_units), len(cols)))
    for i, (r, c) in enumerate(base_units):
        X[i, [0, 1 + regions.index(r), 1 + len(regions) + lins.index(LINEAGE[c]),
              1 + len(regions) + len(lins) + classes.index(c)]] = 1.0
    return X, sd, cols


def main(pkl, out_dir, ct_sd=1.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = pd.read_pickle(pkl)
    names, tx_len, unspliced = z["names"], z["tx_len"], z["unspliced"]
    groups = z["groups"].set_index("group")
    sr_keys = sorted(z["sr"])
    lr_keys = sorted(z["lr"])
    lr_tissue = z["files"].set_index("accession")["tissue"]
    tpm, summary = {}, []

    def frag_models(keys):
        fms = {}
        for i, g in enumerate(keys):
            fm = FragmentModel(tx_len)
            fm.pf = z["sr"][g][1].copy()
            fms[i] = fm
        return fms

    # SR alone: one unit per group
    t0 = time.time()
    em = JointEM([z["sr"][g][0] for g in sr_keys], units=list(range(len(sr_keys))))
    res = fit_bias(em, frag=frag_models(sr_keys), n_outer=8, run_kw=RUN_KW, verbose=True)
    for i, g in enumerate(sr_keys):
        tpm[f"SR_only|{g}"] = molar_to_tpm(res.theta[i])
    summary.append({"fit": "SR_only", "loglik": res.loglik[-1], "n_iter": res.n_iter, "converged": res.converged,
                    "seconds": round(time.time() - t0)})
    print(summary[-1], flush=True)

    # bulk LR alone: one unit per tissue
    t0 = time.time()
    tissues = sorted(set(lr_tissue[lr_keys]))
    em = JointEM([z["lr"][a] for a in lr_keys], units=[tissues.index(lr_tissue[a]) for a in lr_keys])
    tm = AnchoredTruncationModel(tx_len, uniform=unspliced)
    res_lr = fit_bias(em, trunc={i: tm for i in range(len(lr_keys))}, n_outer=8, run_kw=RUN_KW, verbose=True)
    for k, t in enumerate(tissues):
        tpm[f"LR_bulk|{t}"] = molar_to_tpm(res_lr.theta[k])
    summary.append({"fit": "LR_bulk", "loglik": res_lr.loglik[-1], "n_iter": res_lr.n_iter,
                    "converged": res_lr.converged, "seconds": round(time.time() - t0)})
    print(summary[-1], flush=True)

    # joint
    t0 = time.time()
    base_units = sorted({(groups.loc[g, "region"], groups.loc[g, "cell_class"]) for g in sr_keys})
    sr_units = [(groups.loc[g, "region"], groups.loc[g, "cell_class"], groups.loc[g, "age"]) for g in sr_keys]
    units = base_units + sr_units
    K = len(units)
    base = [-1] * len(base_units) + [base_units.index(u[:2]) for u in sr_units]
    scale_groups = [-1] * len(base_units) + [0] * len(sr_units)
    libs = [z["sr"][g][0] for g in sr_keys] + [z["lr"][a] for a in lr_keys]
    pi = np.zeros((len(libs), K))
    pi[np.arange(len(sr_keys)), len(base_units) + np.arange(len(sr_keys))] = 1.0
    composition = {}
    for j, a in enumerate(lr_keys):
        l = len(sr_keys) + j
        allowed = [i for i, (r, _) in enumerate(base_units) if r == lr_tissue[a]]
        pi[l, allowed] = 1.0 / len(allowed)
        composition[l] = allowed
    X, sd, cols = design(base_units, ct_sd)
    em = JointEM(libs, pi=pi, pool="design_gene", design=X, design_sd=sd, base=base, groups=z["gene_groups"],
                 scale_groups=scale_groups, pool_tau=TAU, design_inner=3)
    frag = frag_models(sr_keys)
    tm = AnchoredTruncationModel(tx_len, uniform=unspliced)
    trunc = {len(sr_keys) + j: tm for j in range(len(lr_keys))}
    # warm start: base units from the bulk tissue profile of their region (or uniform), SR units from SR_only
    init = np.zeros((K, len(names)))
    for i, (r, _) in enumerate(base_units):
        init[i] = res_lr.theta[tissues.index(r)] if r in tissues else 1.0
    for i, g in enumerate(sr_keys):
        init[len(base_units) + i] = tpm[f"SR_only|{g}"]
    init = np.maximum(init / init.sum(axis=1, keepdims=True), 1e-10)
    res = fit_bias(em, trunc=trunc, frag=frag, n_outer=8, run_kw={**RUN_KW, "accelerate": "squarem"}, verbose=True,
                   init_theta=init, composition=composition)
    for i, u in enumerate(base_units):
        tpm[f"joint|{u[0]}|{u[1]}"] = molar_to_tpm(res.theta[i])
    for i, u in enumerate(sr_units):
        tpm[f"joint|{u[0]}|{u[1]}|{u[2]}@SR"] = molar_to_tpm(res.theta[len(base_units) + i])
    prof = em.design_profile(X)
    for i, u in enumerate(base_units):
        tpm[f"joint|{u[0]}|{u[1]}|consensus"] = molar_to_tpm(prof[i])
    comp = pd.DataFrame([{"library": lr_keys[j], "tissue": lr_tissue[lr_keys[j]], "unit": f"{base_units[i][0]}|{base_units[i][1]}",
                          "weight": em.pi[len(sr_keys) + j, i]} for j in range(len(lr_keys)) for i in composition[len(sr_keys) + j]])
    comp.to_csv(out / "composition.tsv", sep="\t", index=False)
    print(comp.groupby(["tissue", "unit"]).weight.mean().round(3).to_string(), flush=True)
    summary.append({"fit": "joint", "loglik": res.loglik[-1], "n_iter": res.n_iter, "converged": res.converged,
                    "seconds": round(time.time() - t0)})
    print(summary[-1], flush=True)
    pd.DataFrame(summary).to_csv(out / "fits.tsv", sep="\t", index=False)
    pd.DataFrame(tpm, index=names).to_csv(out / "tpm.tsv.gz", sep="\t", float_format="%.6g")


if __name__ == "__main__":
    main(*sys.argv[1:3], *(float(x) for x in sys.argv[3:4]))
