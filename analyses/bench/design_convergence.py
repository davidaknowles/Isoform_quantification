"""
Convergence benchmark for the design-prior joint fit on one cached problem.

Stage "prepare": build smoke-scale libraries (SR and LR fractions), fit joint_plain, joint_shared and
joint_tpool (transcript pooling) and pickle the libraries and fits.
Stage "variant <name>": load the pickle and fit the design prior from the joint_tpool warm start with one
setting of bias-model warm start / early outer stop, acceleration and design M-step alternations;
writes <dir>/variant_<name>.tsv (MM steps, outer iterations, seconds, final log posterior).

Run: python analyses/bench/design_convergence.py prepare <sr_dirs> <lr_dirs> <ref_dir> <dir> <sr_fraction> <lr_fraction>
     python analyses/bench/design_convergence.py variant <dir> <name>
"""

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parse"))
import joint_celltype as jc  # noqa: E402

CTX_KEYS = ["names", "tx_len", "unspliced", "units", "unit_idx", "gene_groups", "genes"]
# name: (model, copy bias models from the warm start + early outer stop, accelerate, design alternations)
VARIANTS = {"fresh_anderson_x3": ("design", False, "anderson", 3), "warm_anderson_x3": ("design", True, "anderson", 3),
            "warm_anderson_x1": ("design", True, "anderson", 1), "warm_squarem_x1": ("design", True, "squarem", 1),
            "warm_squarem_x3": ("design", True, "squarem", 3),
            "gene_warm_squarem": ("gene", True, "squarem", 1), "gene_warm_anderson": ("gene", True, "anderson", 1),
            "gpool_warm_squarem": ("gpool", True, "squarem", 1), "gpool_warm_anderson": ("gpool", True, "anderson", 1),
            # stopping-rule comparison (theta saved): old single-step rule, 10-iterate window, tight reference
            "warm_anderson_x1_w1": ("design", True, "anderson", 1, {"ll_window": 1}),
            "warm_anderson_x1_w10": ("design", True, "anderson", 1, {"ll_window": 10}),
            "ref_anderson_tight": ("design", True, "anderson", 1, {"ll_window": 20, "ll_rel_tol": 1e-13, "max_iter": 50000}),
            "warm_squarem_x1_save": ("design", True, "squarem", 1, {}),
            # true reference: theta-change rule effectively off, tight log-posterior rule, long cap
            "ref_anderson_true": ("design", True, "anderson", 1, {"rel_tol": 1e-8, "ll_window": 20,
                                                                  "ll_rel_tol": 1e-14, "max_iter": 100000}),
            # finite prior sd on the platform effect (flat by default): anderson default stop vs capped SQUAREM
            **{f"psd{p:g}_{acc}": ("design", True, acc, 1, {}, {"platform": p}) for p in (0.5, 1.0, 2.0)
               for acc in ("anderson", "squarem")}}
SAVE_THETA = {"warm_anderson_x1_w1", "warm_anderson_x1_w10", "ref_anderson_tight", "warm_squarem_x1_save",
              "ref_anderson_true"} | {f"psd{p:g}_{acc}" for p in (0.5, 1.0, 2.0) for acc in ("anderson", "squarem")}


def strip(out):
    return {k: v for k, v in out.items() if k not in ("em", "label")}


def prepare(sr_dirs, kdir, ref_dir, d, sr_fraction, lr_fraction):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    ctx = jc.prepare(sr_dirs, kdir, ref_dir)
    sr_libs, lr_libs, _ = jc.build_libraries(ctx, sr_fraction, lr_fraction)
    both = {**sr_libs, **lr_libs}
    outs = {}
    for fit, use_models, use_length, pool, init in [("joint_plain", False, False, None, None),
                                                    ("joint_shared", True, True, None, "joint_plain"),
                                                    ("joint_tpool", True, True, "transcript", "joint_shared")]:
        t0 = time.time()
        prev = outs.get(init)
        out = jc.fit_one(ctx, both, use_models, use_length, pool=pool, pool_tau=jc.POOL_TAU,
                         init=None if prev is None else (prev["present"], prev["res"].theta))
        outs[fit] = strip(out)
        print(f"{fit}: {out['res'].n_iter} MM steps in final run, {time.time() - t0:.0f} s", flush=True)
    with open(d / "prepared.pkl", "wb") as fh:
        pickle.dump({"ctx": {k: ctx[k] for k in CTX_KEYS}, "libs": both, "outs": outs}, fh, protocol=4)


def variant(d, name):
    d = Path(d)
    with open(d / "prepared.pkl", "rb") as fh:
        z = pickle.load(fh)
    model, warm, acc, inner = VARIANTS[name][:4]
    extra = VARIANTS[name][4] if len(VARIANTS[name]) > 4 else {}
    sd_extra = VARIANTS[name][5] if len(VARIANTS[name]) > 5 else {}
    # design starts from transcript pooling; gene-level fits from the shared joint fit
    tp = z["outs"]["joint_tpool" if model == "design" else "joint_shared"]
    lines = []
    import builtins
    orig = builtins.print
    def capture(*a, **k):                         # count outer iterations and MM steps from fit_bias output
        msg = " ".join(str(x) for x in a)
        if msg.startswith("[bias]"):
            lines.append(msg)
        orig(*a, **k)
    builtins.print = capture
    t0 = time.time()
    kw = dict(pool="design", design_sd={"cell_type": jc.DESIGN_CT_SD, **sd_extra}, em_kw={"design_inner": inner})
    if model == "gene":
        kw = dict(platform_genes=True)
    elif model == "gpool":
        kw = dict(pool="gene")
    out = jc.fit_one(z["ctx"], z["libs"], True, True, init=(tp["present"], tp["res"].theta), pool_tau=jc.POOL_TAU,
                     init_models=tp if warm else None, outer_tol=1e-8 if warm else None,
                     run_kw={"accelerate": acc, **extra}, **kw)
    secs = time.time() - t0
    builtins.print = orig
    steps = sum(int(l.split("n_iter=")[1].split()[0]) for l in lines)
    row = {"variant": name, "model": model, "warm_models": warm, "accelerate": acc, "alternations": inner, "outer": len(lines) - 1,
           "mm_steps_total": steps, "final_run_steps": out["res"].n_iter, "seconds": round(secs),
           "log_posterior": out["res"].loglik[-1], "converged": out["res"].converged}
    print(row, flush=True)
    pd.DataFrame([row]).to_csv(d / f"variant_{name}.tsv", sep="\t", index=False)
    if name in SAVE_THETA:
        import numpy as np
        np.savez(d / f"theta_{name}.npz", theta=out["res"].theta, present=np.array(["@".join(u) for u in out["present"]]))


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(*sys.argv[2:6], float(sys.argv[6]), float(sys.argv[7]))
    else:
        variant(sys.argv[2], sys.argv[3])
