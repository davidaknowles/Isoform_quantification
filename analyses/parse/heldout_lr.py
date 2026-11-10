"""
Held-out long-read likelihood for the Parse SR + Kinnex LR cell-type fits.

A fixed random test_fraction of LR molecules (per molecule, fixed seed) is held out
from every LR library. Fits use a training fraction of the remaining molecules:
  LR     long reads alone, with the read-end model (AnchoredTruncationModel) per kit
  joint  all short reads + training long reads, with all bias models
         (warm-started from the same libraries without bias models)
  joint_gene  as joint, with platform-specific gene totals and within-gene fractions
         shared across platforms (units cell type x platform; warm-started from joint)
  joint_gpool_t<tau>, joint_tpool_t<tau>  (only when requested in [fits]) partial pooling: platform
         gene totals (shared within-gene fractions) or transcript abundances shrunk towards a
         shared cell-type profile, log-scale sd tau (warm-started from joint)
  joint_design_ct<sd>[_tau<tau>]  (only when requested) design prior across cell types: unit log
         abundances = intercept + platform + lineage + cell type (sd <sd>) + unit noise (tau, default 1)
  LR_design_ct<sd>[_tau<tau>]  (only when requested) control: the same design prior on long reads alone
Each held-out LR library (cell type x kit x source library) is scored at the fit's
abundances for its cell type and the fit's bias models for its kit:
ll = sum_e n_e log sum_t mu_t c_et - N log sum_t mu_t b_t, over ECs with read-end
features. Abundances are smoothed with eps of a uniform distribution (two values) so
molecules on transcripts with zero fitted abundance stay finite; their count is reported.
The within-gene score uses molecules whose transcripts all belong to one gene and
conditions on the gene (log sum_{t in e} mu_t c_et - log sum_{t in gene} mu_t b_t), so
gene-level expression differences between donors or platforms do not enter.

Run with several training fractions (separate jobs share the held-out set) to see how
the gain from short reads depends on LR depth.

Writes {out_dir}/heldout.tsv (fit x held-out library), fits.tsv, tpm.tsv.gz and fits.pkl
(abundances and LR bias models per fit, for scoring without refitting).

Run: python analyses/parse/heldout_lr.py <sr_series_dirs> <lr_dirs> <ref_dir> <out_dir>
                                         <lr_train_fraction> [test_fraction] [sr_fraction] [min_cells] [fits]
"""

import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_celltype as jc  # noqa: E402

from joli.joint import JointEM, molar_to_tpm  # noqa: E402
from joli.lr_bias import bias_weights  # noqa: E402

EPS = (1e-6, 1e-9)
# warm-started fits copy the previous fit's bias models and stop the outer loop once it gains less than this
OUTER_TOL = 1e-8


def score(ctx, fit, out, test_libs, genes):
    """Held-out log-likelihood per test library under one fit (units the fit covers)."""
    res, present, label = out["res"], out["present"], out["label"]
    local = {u: i for i, u in enumerate(present)}
    keys = [k for k in sorted(test_libs) if label(k) in local]
    libs = [test_libs[k] for k in keys]
    pi = np.zeros((len(libs), len(present)))
    pi[np.arange(len(libs)), [local[label(k)] for k in keys]] = 1.0
    em = JointEM(libs, pi=pi)
    for l, k in enumerate(keys):
        kit = k.split("|")[2]
        fl = em.libs[l]
        lengths = out["lr_length"] or {}
        c, b = bias_weights(fl.tx, fl.feat, np.ones(len(fl.tx)), fl.b, trunc=out["lr_models"][kit],
                            length=lengths.get(kit))
        em.set_bias(l, c=c, b=b)
    rows = {k: {"fit": fit, "library": k, "unit": k.split("|")[0], "kit": k.split("|")[2],
                "molecules": em.libs[l].N} for l, k in enumerate(keys)}
    for eps in EPS:
        ll, n_zero = em.library_loglik(res.theta, eps=eps)
        llw, used, zw = em.library_loglik_within(res.theta, genes, eps=eps)
        for l, k in enumerate(keys):
            rows[k][f"ll_eps{eps:g}"] = ll[l]
            rows[k]["zero_prob_molecules"] = n_zero[l]
            rows[k][f"ll_within_eps{eps:g}"] = llw[l]
            rows[k][f"within_molecules_eps{eps:g}"] = used[l]
            rows[k]["within_zero_prob_molecules"] = zw[l]
    return pd.DataFrame(rows.values())


def needed(only):
    """Fits to run for the requested ones, including their warm starts: joint_gene and the pooled fits
    start from joint, the design fits from joint_tpool_t1 (closest model), joint from joint_plain."""
    if only is None:
        return None
    need = set(only)
    joint_design = any(f.startswith("joint_design") for f in only)
    if any(f == "joint_gene" or "pool" in f or joint_design or f == "joint" for f in only):
        need |= {"joint", "joint_plain"}
    if joint_design:
        need.add("joint_tpool_t1")
    if any(f.startswith("LR_design") for f in only):
        need.add("LR")
    return need


def main(sr_dirs, kdir, ref_dir, out_dir, lr_train_fraction, test_fraction=0.1, sr_fraction=1.0, min_cells=100,
         only=None):
    """only: comma-separated fits to run (default all); joint_plain always runs as a warm start."""
    only = None if only is None else set(only.split(","))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = jc.prepare(sr_dirs, kdir, ref_dir, min_cells)
    sr_libs, lr_libs, lr_test = jc.build_libraries(ctx, sr_fraction, lr_train_fraction, lr_test_fraction=test_fraction)
    print(f"held out: {len(lr_test)} LR libraries, {sum(l.tcc.ec_counts.sum() for l in lr_test.values()):.0f} molecules",
          flush=True)
    t2g = pd.read_csv(Path(ref_dir) / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    genes = pd.factorize(t2g.reindex(ctx["names"]))[0]                 # -1 where unmapped
    rows, summary, tpm, saved = [], [], {}, {}
    inits, outs = {}, {}
    both = {**sr_libs, **lr_libs}
    configs = [("LR", lr_libs, True, False, False, None, None),
               ("joint_plain", both, False, False, False, None, None),
               ("joint", both, True, True, False, None, "joint_plain"),
               ("joint_gene", both, True, True, True, None, "joint")]
    for f in sorted(needed(only) or [], key=lambda f: ("design" in f, f)):   # design fits after their warm start
        m = re.fullmatch(r"joint_(g|t)pool_t([0-9.]+)", f)
        if m:
            configs.append((f, both, True, True, False, ({"g": "gene", "t": "transcript"}[m.group(1)],
                                                          float(m.group(2))), "joint"))
        m = re.fullmatch(r"joint_design_ct([0-9.]+)(?:_tau([0-9.]+))?", f)
        if m:
            configs.append((f, both, True, True, False, ("design", float(m.group(2) or 1.0),
                                                          {"cell_type": float(m.group(1))}), "joint_tpool_t1"))
        # control: the same design prior across cell types on long reads alone (no SR, no LR length bias)
        m = re.fullmatch(r"LR_design_ct([0-9.]+)(?:_tau([0-9.]+))?", f)
        if m:
            configs.append((f, lr_libs, True, False, False, ("design", float(m.group(2) or 1.0),
                                                              {"cell_type": float(m.group(1))}), "LR"))
    for fit, lib_dict, use_models, use_length, pg, pool, init in configs:
        if only is not None and fit not in needed(only):
            continue
        t0 = time.time()
        out = jc.fit_one(ctx, lib_dict, use_models, use_length, init=inits.get(init), platform_genes=pg,
                         pool=None if pool is None else pool[0], pool_tau=1.0 if pool is None else pool[1],
                         design_sd=pool[2] if pool is not None and len(pool) > 2 else None,
                         init_models=outs.get(init) if use_models else None,
                         outer_tol=OUTER_TOL if outs.get(init) is not None and use_models else None)
        outs[fit] = out if use_models else None
        res = out["res"]
        inits[fit] = (out["present"], res.theta)
        if fit == "joint_plain":
            pass                                     # warm start only: no read-end model, not scored
        else:
            rows.append(score(ctx, fit, out, lr_test, genes))
            saved[fit] = {"present": out["present"], "theta": res.theta, "lr_models": out["lr_models"],
                          "lr_length": out["lr_length"], "platform_genes": pg, "pool": pool}
            for i, u in enumerate(out["present"]):
                tpm[f"{fit}|{'@'.join(u) if isinstance(u, tuple) else u}"] = molar_to_tpm(res.theta[i])
        summary.append({"fit": fit, "lr_train_fraction": float(lr_train_fraction), "n_libraries": len(out["keys"]),
                        "units": len(out["present"]), "loglik": res.loglik[-1], "n_iter": res.n_iter,
                        "converged": res.converged, "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)
    h = pd.concat(rows, ignore_index=True)
    h["lr_train_fraction"] = float(lr_train_fraction)
    h.to_csv(out_dir / "heldout.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "fits.tsv", sep="\t", index=False)
    pd.DataFrame(tpm, index=ctx["names"]).to_csv(out_dir / "tpm.tsv.gz", sep="\t", float_format="%.6g")
    with open(out_dir / "fits.pkl", "wb") as fh:
        pickle.dump({"units": ctx["units"], "fits": saved}, fh)
    wide = h.pivot_table(index="library", columns="fit", values=f"ll_eps{EPS[0]:g}").dropna()
    n = h.drop_duplicates("library").set_index("library")["molecules"].reindex(wide.index)
    for f in [c for c in wide.columns if c != "LR"]:
        if "LR" in wide.columns:
            print(f"held-out LR log-likelihood per molecule, {f} - LR, over {len(wide)} libraries: "
                  f"{(wide[f] - wide['LR']).sum() / n.sum():.4f}", flush=True)


if __name__ == "__main__":
    args = sys.argv[6:]
    conv = [float, float, int, str]
    main(*sys.argv[1:6], *(conv[i](x) for i, x in enumerate(args)))
