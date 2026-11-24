"""
Mouse-brain pilot fits: Tabula Muris Senis Smart-seq2 pseudobulks (region x cell class x age) and ENCODE
bulk PacBio (cortex, hippocampus), from build_libraries.py.

Short reads alone are never fitted at isoform level (they do not identify isoform abundances). Per SR group,
sr_genes.tsv.gz gives gene-level molar profiles from reads in ECs whose targets are all spliced isoforms of
one gene, divided by the gene's effective length (median spliced isoform length minus the mean fragment
length), for the composition deconvolution (composition.py).

Fits:
  LR_bulk  one unit per bulk tissue (AnchoredTruncationModel shared by the LR libraries): bulk LR alone
  joint    base units (region, class) carry transcript abundances with a design prior (intercept + region
           + lineage + class effects, unit noise TAU); each SR group is its base unit rescaled per gene
           (design_gene) with one per-gene SR scale profile shared by all SR units (scale_groups), so SR
           gene expression informs the base units; each bulk LR library is a mixture over its region's base
           units, with composition estimated in the EM (fit_bias composition) or, when a composition table
           from composition.py (gene-level deconvolution) is given, fixed at it as read shares (JointEM
           mixture="reads"; the deconvolution gives molecule shares, taken as read shares, i.e. similar
           mean long-read sampling rates across cell classes). Base units of regions without
           bulk LR (striatum, cerebellum) are informed by SR and the prior.
Outputs: tpm.tsv.gz (transcripts x fit|unit; joint base units "<region>|<class>", SR units
"<region>|<class>|<age>@SR", consensus profiles "<region>|<class>|consensus"), sr_genes.tsv.gz (genes, indexed by
one spliced isoform each, x SR group), composition.tsv, fits.tsv.

Options as key=value after the positional arguments: tau (unit noise sd, default TAU), rsd and lsd (region and
lineage effect sds, default REGION_SD, LINEAGE_SD) and srw (SR counts multiplied by srw, tempering the short-read
likelihood against the long reads; default SR_WEIGHT), lr_test (a fraction of each bulk long-read library's molecules held
out by binomial thinning, fixed seed; default 0) and holdout_region (all bulk long-read libraries of that region left
out of every fit and scored afterwards: leave-one-region-out), holdout_donor (the same for all bulk libraries of one
donor, files.tsv donor column) and composition_by (tissue: one composition per tissue from composition.tsv, the
default; library: each bulk library's composition by gene-level deconvolution of its own single-gene-EC long-read counts
against the SR gene profiles of its tissue's classes, so libraries of one tissue with different cell mixtures, e.g.
heart chambers, get their own weights). Strong class shrinkage (small ct_sd and lsd) and srw < 1 were needed
on PBMC (analyses/pbmc_ss2/predict_kinnex.py).

With lr_test or holdout_region, the held-out molecules are scored under the joint fit (each library's mixture of its
tissue's base units with the composition used in fitting, the fitted read-end model) and under LR_bulk (the
library's tissue profile, or the mean of the fitted tissue profiles for a held-out region); per-EC log-probabilities (all, and within gene) go to
heldout_lr.npz for select_srw.py, which compares runs (e.g. a grid of srw) on ECs every run can score.

Run: python analyses/mouse_brain/fit.py <libraries.pkl> <out_dir> [ct_sd] [composition.tsv | -] [key=value ...]
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.deconv import mixture_weights, single_gene_counts
from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import AnchoredTruncationModel, FragmentModel, bias_weights, fit_bias
from joli.paf import split_library

LINEAGE = {"neuron": "neuron", "astrocyte": "glia", "oligodendrocyte": "glia", "OPC": "glia", "microglia": "immune",
           "endothelial": "vascular", "pericyte": "vascular"}
TAU = 1.0
# short-read count weight: tempering SR against the long reads; 0.3 had the best agreement with the single-cell long-read
# truth in the mouse-brain weight grids (held-out bulk long reads do not select it; LABNOTEBOOK 2026-11-21/22)
SR_WEIGHT = 0.3
REGION_SD, LINEAGE_SD = 2.0, 2.0
RUN_KW = dict(max_iter=20000, ll_rel_tol=1e-10)


def design(base_units, ct_sd, region_sd=REGION_SD, lineage_sd=LINEAGE_SD):
    """Design rows of (region, class) base units: intercept, region, lineage and class indicators."""
    regions = sorted({r for r, _ in base_units})
    lins = sorted(set(LINEAGE.values()))
    classes = sorted({c for _, c in base_units})
    cols = ["intercept"] + [f"region:{r}" for r in regions] + [f"lineage:{l}" for l in lins] + [f"class:{c}" for c in classes]
    sd = np.r_[np.inf, np.full(len(regions), region_sd), np.full(len(lins), lineage_sd), np.full(len(classes), ct_sd)]
    X = np.zeros((len(base_units), len(cols)))
    for i, (r, c) in enumerate(base_units):
        X[i, [0, 1 + regions.index(r), 1 + len(regions) + lins.index(LINEAGE[c]),
              1 + len(regions) + len(lins) + classes.index(c)]] = 1.0
    return X, sd, cols


def sr_genes(z, keys):
    """Gene-level molar profiles of SR groups (genes indexed by their first spliced isoform; see module doc)."""
    names, tx_len = np.asarray(z["names"]), z["tx_len"]
    gene = np.where(z["unspliced"], -1, z["genes"])
    sp = gene >= 0
    first = pd.Series(np.flatnonzero(sp)).groupby(gene[sp]).first()           # gene code -> first spliced isoform
    glen = pd.Series(tx_len[sp]).groupby(gene[sp]).median().to_numpy()
    fmid = FragmentModel(tx_len[:1]).fmid
    out = {}
    for g in keys:
        lib, pf = z["sr"][g]
        ecs = lib.tcc.ec_transcripts
        n = single_gene_counts(ecs.ptr, ecs.values, lib.tcc.ec_counts, gene, int(gene.max()) + 1)[first.index]
        x = n / np.maximum(glen - pf @ fmid + 1.0, 1.0)
        out[g] = x / x.sum()
    return pd.DataFrame(out, index=pd.Index(names[first.to_numpy()], name="gene_isoform"))


def scaled(lib, w):
    """Library with EC counts multiplied by w (w = 1: the library itself)."""
    if w == 1.0:
        return lib
    tcc = copy.copy(lib.tcc)
    tcc.ec_counts = np.asarray(lib.tcc.ec_counts, dtype=np.float64) * w
    out = copy.copy(lib)
    out.tcc = tcc
    return out


def library_compositions(z, srg, groups, sr_keys, base_units, lr_tissue, keys):
    """
    {library: {base unit: weight}} by gene-level deconvolution of each bulk library: reads in its ECs whose targets
    are all spliced isoforms of one gene, against the mean SR gene profile of each class of its tissue (srg columns).
    """
    gene = np.where(z["unspliced"], -1, z["genes"])
    sp = gene >= 0
    first = pd.Series(np.flatnonzero(sp)).groupby(gene[sp]).first()
    out = {}
    for a in keys:
        units = [u for u in base_units if u[0] == lr_tissue[a]]
        ref = np.stack([srg[[g for g in sr_keys if (groups.loc[g, "region"], groups.loc[g, "cell_class"]) == u]].mean(axis=1)
                        .to_numpy() for u in units])
        ecs = z["lr"][a].tcc.ec_transcripts
        y = single_gene_counts(ecs.ptr, ecs.values, z["lr"][a].tcc.ec_counts, gene, int(gene.max()) + 1)[first.index]
        keep = ref.sum(axis=0) > 0
        w, _ = mixture_weights(y[keep], ref[:, keep])
        out[a] = dict(zip(units, w))
        print(f"composition {a} ({lr_tissue[a]}): " + ", ".join(f"{u[1]} {v:.3f}" for u, v in zip(units, w)), flush=True)
    return out


def heldout_logprobs(libs, pi, theta, tm, gene_groups, mixture):
    """Per-EC log-probabilities (all, within gene) and counts of held-out libraries at fixed abundances."""
    em = JointEM(libs, pi=pi, mixture=mixture)
    for l, fl in enumerate(em.libs):
        c, b = bias_weights(fl.tx, fl.feat, np.ones(len(fl.tx)), fl.b, trunc=tm)
        em.set_bias(l, c=c, b=b)
    return em.ec_logprob(theta), em.ec_logprob(theta, groups=gene_groups), [fl.counts for fl in em.libs]


def main(pkl, out_dir, ct_sd=1.0, composition_tsv=None, tau=TAU, rsd=REGION_SD, lsd=LINEAGE_SD, srw=SR_WEIGHT, lr_test=0.0,
         holdout_region=None, holdout_donor=None, composition_by="tissue"):
    if composition_tsv == "-":
        composition_tsv = None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = pd.read_pickle(pkl)
    names, tx_len, unspliced = z["names"], z["tx_len"], z["unspliced"]
    groups = z["groups"].set_index("group")
    sr_keys = sorted(z["sr"])
    lr_tissue = z["files"].set_index("accession")["tissue"]
    lr_donor = z["files"].set_index("accession")["donor"]
    held = lambda a: lr_tissue[a] == holdout_region or lr_donor[a] == holdout_donor
    lr_keys = sorted(a for a in z["lr"] if not held(a))
    tpm, summary = {}, []
    test = {a: z["lr"][a] for a in sorted(z["lr"]) if held(a)}
    if (holdout_region is not None or holdout_donor is not None) and not test:
        raise ValueError("no bulk long-read libraries to hold out")
    if float(lr_test) > 0:
        for j, a in enumerate(lr_keys):
            z["lr"][a], test[a] = split_library(z["lr"][a], float(lr_test), seed=j)

    def frag_models(keys):
        fms = {}
        for i, g in enumerate(keys):
            fm = FragmentModel(tx_len)
            fm.pf = z["sr"][g][1].copy()
            fms[i] = fm
        return fms

    # SR gene-level profiles (no isoform-level estimation)
    srg = sr_genes(z, sr_keys)
    srg.to_csv(out / "sr_genes.tsv.gz", sep="\t", float_format="%.6g")

    # bulk LR alone: one unit per tissue
    t0 = time.time()
    tissues = sorted(set(lr_tissue[lr_keys]))
    em = JointEM([z["lr"][a] for a in lr_keys], units=[tissues.index(lr_tissue[a]) for a in lr_keys])
    tm_lr = AnchoredTruncationModel(tx_len, uniform=unspliced)
    res_lr = fit_bias(em, trunc={i: tm_lr for i in range(len(lr_keys))}, n_outer=8, run_kw=RUN_KW, verbose=True)
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
    libs = [scaled(z["sr"][g][0], float(srw)) for g in sr_keys] + [z["lr"][a] for a in lr_keys]
    pi = np.zeros((len(libs), K))
    pi[np.arange(len(sr_keys)), len(base_units) + np.arange(len(sr_keys))] = 1.0
    composition = {}
    fixed = None
    if composition_tsv is not None:
        fixed = pd.read_csv(composition_tsv, sep="\t").set_index("unit")["weight"]
    libcomp = {}
    if composition_by == "library":
        libcomp = library_compositions(z, srg, groups, sr_keys, base_units, lr_tissue, list(lr_keys) + sorted(test))
        pd.DataFrame([{"library": a, "tissue": lr_tissue[a], "unit": f"{u[0]}|{u[1]}", "weight": w}
                      for a, d in libcomp.items() for u, w in d.items()]).to_csv(out / "library_composition.tsv",
                                                                                 sep="\t", index=False)
        fixed = True                                     # fixed weights (read shares), no EM composition
    mode = "molecules" if fixed is None else "reads"

    def comp_row(a):
        """Mixture weights over all units of bulk library a: its region's base units (composition or uniform)."""
        row = np.zeros(K)
        allowed = [i for i, (r, _) in enumerate(base_units) if r == lr_tissue[a]]
        if libcomp:
            row[allowed] = [libcomp[a].get(base_units[i], 0.0) for i in allowed]
            row /= row.sum()
        elif fixed is None:
            row[allowed] = 1.0 / len(allowed)
        else:
            w = fixed.reindex([f"{r}|{c}" for r, c in (base_units[i] for i in allowed)]).fillna(0.0).to_numpy()
            row[allowed] = w / w.sum()
        return row, allowed

    for j, a in enumerate(lr_keys):
        l = len(sr_keys) + j
        pi[l], composition[l] = comp_row(a)
    X, sd, cols = design(base_units, ct_sd, float(rsd), float(lsd))
    em = JointEM(libs, pi=pi, pool="design_gene", design=X, design_sd=sd, base=base, groups=z["gene_groups"],
                 scale_groups=scale_groups, pool_tau=float(tau), design_inner=3,
                 mixture=mode)
    frag = frag_models(sr_keys)
    tm = AnchoredTruncationModel(tx_len, uniform=unspliced)
    trunc = {len(sr_keys) + j: tm for j in range(len(lr_keys))}
    # warm start: base units from the bulk tissue profile of their region (the mean bulk profile for regions
    # without bulk long reads); SR units from their base unit
    init = np.zeros((K, len(names)))
    for i, (r, _) in enumerate(base_units):
        init[i] = res_lr.theta[tissues.index(r)] if r in tissues else res_lr.theta.mean(axis=0)
    for i, u in enumerate(sr_units):
        init[len(base_units) + i] = init[base_units.index(u[:2])]
    init = np.maximum(init / init.sum(axis=1, keepdims=True), 1e-10)
    res = fit_bias(em, trunc=trunc, frag=frag, n_outer=8, run_kw={**RUN_KW, "accelerate": "squarem"}, verbose=True,
                   init_theta=init, composition=composition if fixed is None else None)
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
    if test:
        tkeys = sorted(test)
        tlibs = [test[a] for a in tkeys]
        # the composition used in fitting (estimated by the EM for trained libraries without a fixed composition)
        trained = {a: em.pi[len(sr_keys) + j] for j, a in enumerate(lr_keys)}
        tpi = np.stack([trained[a] if a in trained else comp_row(a)[0] for a in tkeys])
        jb = heldout_logprobs(tlibs, tpi, res.theta, tm, z["gene_groups"], mode)
        btheta = np.stack([res_lr.theta[tissues.index(lr_tissue[a])] if lr_tissue[a] in tissues
                           else res_lr.theta.mean(axis=0) for a in tkeys])
        lb = heldout_logprobs(tlibs, np.eye(len(tkeys)), btheta, tm_lr, z["gene_groups"], "molecules")
        arrays = {}
        for j, a in enumerate(tkeys):
            arrays[f"{a}|counts"] = jb[2][j]
            for name, (all_, within, _) in (("joint", jb), ("LR_bulk", lb)):
                arrays[f"{a}|{name}|all"] = all_[j]
                arrays[f"{a}|{name}|within_gene"] = within[j]
        np.savez_compressed(out / "heldout_lr.npz", **arrays)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if "=" not in a]
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    main(*pos[:2], *(float(x) for x in pos[2:3]), *pos[3:4], **kw)
