"""
How well do Smart-seq2 cell-type pseudobulks plus bulk long reads predict cell-type-resolved long reads?
The atlas setting (short-read cell types + bulk tissue long reads, composition by gene-level deconvolution)
on PBMC, where the long reads carry cell labels to check against.

Long reads: 10x Kinnex PBMC molecules (analyses/kinnex_10x); a fixed per-molecule test_fraction is held out
(joint_celltype.lr_libraries, as heldout_lr.py) and lr_fraction of the rest is used for training. Training
molecules of all labelled cells are pooled per kit x source library into label-free bulk libraries.
Short reads: Smart-seq2 PBMC groups (cell type x donor; make_groups.py, build_libraries.py), other donors.
Cell types are merged into classes both data sets cover (CLASS); held-out molecules are scored per class.

Fits:
  LR_bulk      one abundance vector per bulk library, read-end model per kit: no cell-type resolution
  joint        base units per class with the design prior (intercept + lineage + class); each Smart-seq2
               group is its class unit rescaled per gene (one shared SR scale profile); each bulk library is
               a mixture over the classes with read shares fixed by gene-level deconvolution (bulk LR gene
               profile against Smart-seq2 gene profiles from single-gene ECs); joint_consensus is the design
               profile of each class
  LR_celltype  the training long reads with their cell labels, one unit per class (the ceiling)
Composition: deconvolution estimates are reported next to the training molecules' label shares, which are
not used in fitting.

Scores (score.tsv): per held-out library (cell type x kit x source library), log-likelihood per molecule of
each fit's profile for its class (and the source library's profile for LR_bulk) under that fit's read-end
model for the kit, without smoothing, on ECs with positive probability under every fit; "all" and
"within_gene" (ECs of one gene, conditioned on the gene). summary.tsv: gain per molecule over LR_bulk.

Run: python analyses/pbmc_ss2/predict_kinnex.py <kinnex_dir> <ss2_libraries.pkl> <ref_dir> <out_dir>
                                                 [lr_fraction] [test_fraction]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parse"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mouse_brain"))
import fit as atlas  # noqa: E402
import joint_celltype as jc  # noqa: E402

from joli.deconv import mixture_weights  # noqa: E402
from joli.joint import JointEM, molar_to_tpm  # noqa: E402
from joli.lr_bias import AnchoredTruncationModel, FragmentModel, bias_weights, fit_bias  # noqa: E402
from joli.paf import concat_libraries, read_fasta_lengths  # noqa: E402

CLASS = {"Tcm/Naive_helper_T_cells": "CD4_T", "Tem/Effector_helper_T_cells": "CD4_T", "Regulatory_T_cells": "CD4_T",
         "Tem/Temra_cytotoxic_T_cells": "CD8_T", "Tem/Trm_cytotoxic_T_cells": "CD8_T",
         "Tcm/Naive_cytotoxic_T_cells": "CD8_T", "CD8a/a": "CD8_T", "MAIT_cells": "CD8_T",
         "CRTAM+_gamma-delta_T_cells": "CD8_T", "CD16+_NK_cells": "NK", "CD16-_NK_cells": "NK", "NK_cells": "NK",
         "Naive_B_cells": "B", "Memory_B_cells": "B", "Age-associated_B_cells": "B",
         "Classical_monocytes": "Mono_classical", "Non-classical_monocytes": "Mono_nonclassical"}
LINEAGE = {"CD4_T": "T", "CD8_T": "T", "NK": "NK", "B": "B", "Mono_classical": "myeloid", "Mono_nonclassical": "myeloid"}
LINEAGE_SD, CLASS_SD, TAU = 2.0, 0.5, 1.0
RUN_KW = dict(max_iter=20000, ll_rel_tol=1e-10)


def design(classes):
    lins = sorted(set(LINEAGE.values()))
    X = np.zeros((len(classes), 1 + len(lins) + len(classes)))
    for i, c in enumerate(classes):
        X[i, [0, 1 + lins.index(LINEAGE[c]), 1 + len(lins) + i]] = 1.0
    sd = np.r_[np.inf, np.full(len(lins), LINEAGE_SD), np.full(len(classes), CLASS_SD)]
    return X, sd


def kit_models(keys, kit_of, tx_len, unspliced):
    """One AnchoredTruncationModel per kit, shared by that kit's libraries: {library index: model}."""
    models = {k: AnchoredTruncationModel(tx_len, uniform=unspliced) for k in sorted(set(kit_of(x) for x in keys))}
    return {i: models[kit_of(k)] for i, k in enumerate(keys)}, models


def gene_profile(theta, genes, first):
    """Molar gene profiles (rows of theta summed over spliced isoforms per gene), indexed like sr_genes."""
    out = np.stack([np.bincount(genes[genes >= 0], weights=t[genes >= 0], minlength=int(genes.max()) + 1) for t in theta])
    return out[:, first.index]


def logprobs(lib, theta, model, genes):
    em = JointEM([lib], units=[0])
    fl = em.libs[0]
    c, b = bias_weights(fl.tx, fl.feat, np.ones(len(fl.tx)), fl.b, trunc=model)
    em.set_bias(0, c=c, b=b)
    return em.ec_logprob(theta[None]), em.ec_logprob(theta[None], groups=genes), fl.counts


def main(kdir, ss2_pkl, ref_dir, out_dir, lr_fraction=0.25, test_fraction=0.1):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(ref_dir)
    names, tx_len = read_fasta_lengths(ref_dir / "spliceu.fa")
    tx_len = np.asarray(tx_len)
    unspliced = np.array([n.endswith("-U") for n in names])
    t2g = pd.read_csv(ref_dir / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    genes = pd.factorize(t2g.reindex(names))[0]
    n_genes = int(genes.max()) + 1
    gene_groups = np.where(genes < 0, -1, genes + np.where(unspliced, n_genes, 0))
    spliced_gene = np.where(unspliced, -1, genes)

    # long reads: all labelled cells' groups for training, held-out molecules per class
    src = jc.lr_sources([kdir])
    labels = {g for _, gs, _ in src for g in gs if g is not None}
    rates = {g: jc.primer_rates(g.split("|")[2].split("@")[0], tx_len, 0, unspliced, kappa=jc.KAPPA0) for g in labels}
    t0 = time.time()
    lr_train, lr_test = jc.lr_libraries(src, names, float(lr_fraction), rates,
                                        AnchoredTruncationModel(tx_len, uniform=unspliced), test_fraction=float(test_fraction))
    print(f"LR: {len(lr_train)} training and {len(lr_test)} held-out libraries ({time.time() - t0:.0f} s)", flush=True)
    kit_of = lambda k: k.split("|")[2]
    bulk_of = lambda k: "bulk|LR|" + "|".join(k.split("|")[2:])
    bulk = {}
    for k in sorted(lr_train):
        bulk.setdefault(bulk_of(k), []).append(lr_train[k])
    bulk = {b: concat_libraries(b, ls) for b, ls in bulk.items()}
    bulk_keys = sorted(bulk)
    true_comp = pd.DataFrame([{"library": bulk_of(k), "class": CLASS.get(k.split("|")[0], "other"), "molecules": l.tcc.ec_counts.sum()}
                              for k, l in lr_train.items()]).groupby(["library", "class"]).molecules.sum()
    true_comp = (true_comp / true_comp.groupby(level=0).transform("sum")).rename("true")

    # short reads
    z = pd.read_pickle(ss2_pkl)
    if list(z["names"]) != list(names):
        raise ValueError("Smart-seq2 libraries use a different reference order")
    groups = z["groups"].set_index("group")
    sr_keys = sorted(g for g in z["sr"] if CLASS.get(groups.loc[g, "cell_class"]) is not None)
    sr_class = [CLASS[groups.loc[g, "cell_class"]] for g in sr_keys]
    classes = sorted(set(sr_class))
    print(f"classes {classes}; {len(sr_keys)} Smart-seq2 groups", flush=True)
    summary, tpm, fits = [], {}, {}

    def record(name, res, t0, n_libs):
        summary.append({"fit": name, "libraries": n_libs, "loglik": res.loglik[-1], "n_iter": res.n_iter,
                        "converged": res.converged, "seconds": round(time.time() - t0)})
        print(summary[-1], flush=True)

    # LR_bulk
    t0 = time.time()
    em = JointEM([bulk[b] for b in bulk_keys], units=list(range(len(bulk_keys))))
    trunc, models = kit_models(bulk_keys, kit_of, tx_len, unspliced)
    res = fit_bias(em, trunc=trunc, n_outer=8, run_kw=RUN_KW, verbose=True)
    record("LR_bulk", res, t0, len(bulk_keys))
    fits["LR_bulk"] = ({b: res.theta[i] for i, b in enumerate(bulk_keys)}, models)
    for i, b in enumerate(bulk_keys):
        tpm[f"LR_bulk|{b}"] = molar_to_tpm(res.theta[i])
    lr_bulk_theta = res.theta

    # LR_celltype (labelled training long reads, the ceiling)
    t0 = time.time()
    keys = sorted(k for k in lr_train if k.split("|")[0] in CLASS and CLASS[k.split("|")[0]] in classes)
    em = JointEM([lr_train[k] for k in keys], units=[classes.index(CLASS[k.split("|")[0]]) for k in keys])
    trunc, models = kit_models(keys, kit_of, tx_len, unspliced)
    res = fit_bias(em, trunc=trunc, n_outer=8, run_kw=RUN_KW, verbose=True)
    record("LR_celltype", res, t0, len(keys))
    fits["LR_celltype"] = ({c: res.theta[i] for i, c in enumerate(classes)}, models)
    for i, c in enumerate(classes):
        tpm[f"LR_celltype|{c}"] = molar_to_tpm(res.theta[i])

    # composition by gene-level deconvolution
    srg = atlas.sr_genes(z, sr_keys)
    first = pd.Series(np.flatnonzero(spliced_gene >= 0)).groupby(spliced_gene[spliced_gene >= 0]).first()
    ref = np.stack([srg[[g for g, c in zip(sr_keys, sr_class) if c == cl]].mean(axis=1).to_numpy() for cl in classes])
    bulk_genes = gene_profile(lr_bulk_theta, spliced_gene, first)
    keep = ref.sum(axis=0) > 0
    comp_rows, comp = [], {}
    for i, b in enumerate(bulk_keys):
        y = bulk_genes[i, keep]
        w, _ = mixture_weights(y / y.sum() * 1e6, ref[:, keep])
        comp[b] = w
        for c, v in zip(classes, w):
            comp_rows.append({"library": b, "class": c, "deconvolution": v, "true": true_comp.get((b, c), 0.0)})
        comp_rows.append({"library": b, "class": "other", "deconvolution": 0.0, "true": true_comp.get((b, "other"), 0.0)})
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out / "composition.tsv", sep="\t", index=False)
    print(comp_df.groupby("class")[["deconvolution", "true"]].mean().round(3).to_string(), flush=True)

    # joint: Smart-seq2 groups + bulk long reads
    t0 = time.time()
    nb = len(classes)
    K = nb + len(sr_keys)
    base = [-1] * nb + [classes.index(c) for c in sr_class]
    libs = [z["sr"][g][0] for g in sr_keys] + [bulk[b] for b in bulk_keys]
    pi = np.zeros((len(libs), K))
    pi[np.arange(len(sr_keys)), nb + np.arange(len(sr_keys))] = 1.0
    for j, b in enumerate(bulk_keys):
        pi[len(sr_keys) + j, :nb] = comp[b] / comp[b].sum()
    X, sd = design(classes)
    em = JointEM(libs, pi=pi, pool="design_gene", design=X, design_sd=sd, base=base, groups=gene_groups,
                 scale_groups=[-1] * nb + [0] * len(sr_keys), pool_tau=TAU, design_inner=3, mixture="reads")
    frag = {}
    for i, g in enumerate(sr_keys):
        fm = FragmentModel(tx_len)
        fm.pf = z["sr"][g][1].copy()
        frag[i] = fm
    trunc_b, models = kit_models(bulk_keys, kit_of, tx_len, unspliced)
    trunc = {len(sr_keys) + j: m for j, m in trunc_b.items()}
    init = np.tile(lr_bulk_theta.mean(axis=0), (K, 1))
    init = np.maximum(init / init.sum(axis=1, keepdims=True), 1e-10)
    res = fit_bias(em, trunc=trunc, frag=frag, n_outer=8, run_kw={**RUN_KW, "accelerate": "squarem"}, verbose=True,
                   init_theta=init)
    record("joint", res, t0, len(libs))
    prof = em.design_profile(X)
    fits["joint"] = ({c: res.theta[i] for i, c in enumerate(classes)}, models)
    fits["joint_consensus"] = ({c: prof[i] for i, c in enumerate(classes)}, models)
    for i, c in enumerate(classes):
        tpm[f"joint|{c}"] = molar_to_tpm(res.theta[i])
        tpm[f"joint_consensus|{c}"] = molar_to_tpm(prof[i])
    pd.DataFrame(summary).to_csv(out / "fits.tsv", sep="\t", index=False)
    pd.DataFrame(tpm, index=names).to_csv(out / "tpm.tsv.gz", sep="\t", float_format="%.6g")

    # score held-out long reads per class
    rows = []
    for k in sorted(lr_test):
        c = CLASS.get(k.split("|")[0])
        if c not in classes:
            continue
        lp = {}
        for name, (profiles, models) in fits.items():
            theta = profiles[bulk_of(k)] if name == "LR_bulk" else profiles[c]
            lp[name] = logprobs(lr_test[k], theta, models[kit_of(k)], gene_groups)
        counts = lp["LR_bulk"][2]
        for score, idx in (("all", 0), ("within_gene", 1)):
            ok = np.logical_and.reduce([np.isfinite(v[idx][0]) for v in lp.values()])
            for name, v in lp.items():
                rows.append({"library": k, "class": c, "kit": kit_of(k), "score": score, "fit": name,
                             "molecules": counts[ok].sum(), "ll": counts[ok] @ v[idx][0][ok],
                             "scorable": counts[np.isfinite(lp["LR_bulk"][idx][0])].sum()})
    r = pd.DataFrame(rows)
    r.to_csv(out / "score.tsv", sep="\t", index=False)
    base_ll = r[r.fit == "LR_bulk"].set_index(["library", "score"]).ll
    r["gain"] = r.ll - base_ll.reindex(pd.MultiIndex.from_frame(r[["library", "score"]])).to_numpy()
    s = r.groupby(["score", "fit", "class"]).agg(gain=("gain", "sum"), molecules=("molecules", "sum")).reset_index()
    s["gain_per_molecule"] = s.gain / s.molecules
    tot = r.groupby(["score", "fit"]).agg(gain=("gain", "sum"), molecules=("molecules", "sum")).reset_index()
    tot["gain_per_molecule"] = tot.gain / tot.molecules
    tot["class"] = "all"
    s = pd.concat([s, tot], ignore_index=True)
    s.to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(s.pivot_table(index=["score", "class"], columns="fit", values="gain_per_molecule").round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5], *(float(x) for x in sys.argv[5:7]))
