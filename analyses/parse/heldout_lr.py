"""
Held-out long-read likelihood for the Parse SR + Kinnex LR cell-type fits.

A fixed random test_fraction of LR molecules (per molecule, fixed seed) is held out
from every LR library. Fits use a training fraction of the remaining molecules:
  LR     long reads alone, with the read-end model (AnchoredTruncationModel) per kit
  joint  all short reads + training long reads, with all bias models
         (warm-started from the same libraries without bias models)
Each held-out LR library (cell type x kit x source library) is scored at the fit's
abundances for its cell type and the fit's bias models for its kit:
ll = sum_e n_e log sum_t mu_t c_et - N log sum_t mu_t b_t, over ECs with read-end
features. Abundances are smoothed with eps of a uniform distribution (two values) so
molecules on transcripts with zero fitted abundance stay finite; their count is reported.

Run with several training fractions (separate jobs share the held-out set) to see how
the gain from short reads depends on LR depth.

Writes {out_dir}/heldout.tsv (fit x held-out library), fits.tsv, tpm.tsv.gz.

Run: python analyses/parse/heldout_lr.py <sr_series_dirs> <lr_dirs> <ref_dir> <out_dir>
                                         <lr_train_fraction> [test_fraction] [sr_fraction] [min_cells]
"""

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


def score(ctx, fit, out, test_libs):
    """Held-out log-likelihood per test library under one fit (units the fit covers)."""
    res, present = out["res"], out["present"]
    local = {k: i for i, k in enumerate(present)}
    keys = [k for k in sorted(test_libs) if ctx["unit_idx"][k.split("|")[0]] in local]
    libs = [test_libs[k] for k in keys]
    pi = np.zeros((len(libs), len(present)))
    pi[np.arange(len(libs)), [local[ctx["unit_idx"][k.split("|")[0]]] for k in keys]] = 1.0
    em = JointEM(libs, pi=pi, unit_names=[ctx["units"][k] for k in present])
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
        for l, k in enumerate(keys):
            rows[k][f"ll_eps{eps:g}"] = ll[l]
            rows[k]["zero_prob_molecules"] = n_zero[l]
    return pd.DataFrame(rows.values())


def main(sr_dirs, kdir, ref_dir, out_dir, lr_train_fraction, test_fraction=0.1, sr_fraction=1.0, min_cells=100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = jc.prepare(sr_dirs, kdir, ref_dir, min_cells)
    sr_libs, lr_libs, lr_test = jc.build_libraries(ctx, sr_fraction, lr_train_fraction, lr_test_fraction=test_fraction)
    print(f"held out: {len(lr_test)} LR libraries, {sum(l.tcc.ec_counts.sum() for l in lr_test.values()):.0f} molecules",
          flush=True)
    rows, summary, tpm = [], [], {}
    plain = None
    for fit, lib_dict, use_models, use_length in [("LR", lr_libs, True, False),
                                                   ("joint_plain", {**sr_libs, **lr_libs}, False, False),
                                                   ("joint", {**sr_libs, **lr_libs}, True, True)]:
        t0 = time.time()
        out = jc.fit_one(ctx, lib_dict, use_models, use_length, init=plain if fit == "joint" else None)
        res = out["res"]
        if fit == "joint_plain":
            plain = (out["present"], res.theta)      # warm start only: no read-end model, not scored
        else:
            rows.append(score(ctx, fit, out, lr_test))
            for i, k in enumerate(out["present"]):
                tpm[f"{fit}|{ctx['units'][k]}"] = molar_to_tpm(res.theta[i])
        summary.append({"fit": fit, "lr_train_fraction": float(lr_train_fraction), "n_libraries": len(out["keys"]),
                        "units": len(out["present"]), "loglik": res.loglik[-1], "n_iter": res.n_iter,
                        "converged": res.converged, "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)
    h = pd.concat(rows, ignore_index=True)
    h["lr_train_fraction"] = float(lr_train_fraction)
    h.to_csv(out_dir / "heldout.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "fits.tsv", sep="\t", index=False)
    pd.DataFrame(tpm, index=ctx["names"]).to_csv(out_dir / "tpm.tsv.gz", sep="\t", float_format="%.6g")
    both = h.pivot_table(index="library", columns="fit", values=f"ll_eps{EPS[0]:g}").dropna()
    n = h.drop_duplicates("library").set_index("library")["molecules"].reindex(both.index)
    print(f"held-out LR log-likelihood per molecule, joint - LR, over {len(both)} libraries: "
          f"{(both['joint'] - both['LR']).sum() / n.sum():.4f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6], *(float(x) if i < 2 else int(x) for i, x in enumerate(sys.argv[6:9])))
