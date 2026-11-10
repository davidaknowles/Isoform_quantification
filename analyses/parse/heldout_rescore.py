"""
Rescore saved held-out fits on molecules with nonzero probability under every fit.

heldout_lr.py smooths abundances with eps so that held-out LR molecules on transcripts
with zero fitted abundance stay finite; fits that leave fewer such molecules (e.g. the
design prior, which gives every transcript a shrunk abundance) then gain partly through
the choice of eps. Here every fit saved in <heldout_dir>/train_<f>*/fits.pkl is scored
without smoothing on the held-out ECs that have positive probability under all fits of
that training fraction: per-read log-probability (all) and given the gene (within_gene,
single-gene ECs). The held-out libraries (same seed and test fraction as heldout_lr.py)
are built once.

Writes <heldout_dir>/rescore.tsv (fraction x fit x score x library) and rescore_summary.tsv.

Run: python analyses/parse/heldout_rescore.py <sr_series_dirs> <lr_dirs> <ref_dir> <heldout_dir> [test_fraction] [min_cells]
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_celltype as jc  # noqa: E402

from joli.joint import JointEM  # noqa: E402
from joli.lr_bias import AnchoredTruncationModel, bias_weights  # noqa: E402


def label_fn(present):
    if present and isinstance(present[0], tuple):
        return lambda k: (k.split("|")[0], k.split("|")[1])
    return lambda k: k.split("|")[0]


def fit_logprobs(fit, test_libs, keys, genes):
    """Per-EC log-probabilities (all, within gene) of the held-out libraries `keys` under a saved fit."""
    present, label = fit["present"], label_fn(fit["present"])
    local = {u: i for i, u in enumerate(present)}
    libs = [test_libs[k] for k in keys]
    pi = np.zeros((len(libs), len(present)))
    pi[np.arange(len(libs)), [local[label(k)] for k in keys]] = 1.0
    em = JointEM(libs, pi=pi)
    lengths = fit["lr_length"] or {}
    for l, k in enumerate(keys):
        kit = k.split("|")[2]
        fl = em.libs[l]
        c, b = bias_weights(fl.tx, fl.feat, np.ones(len(fl.tx)), fl.b, trunc=fit["lr_models"][kit],
                            length=lengths.get(kit))
        em.set_bias(l, c=c, b=b)
    return em.ec_logprob(fit["theta"]), em.ec_logprob(fit["theta"], groups=genes), [fl.counts for fl in em.libs]


def main(sr_dirs, kdir, ref_dir, heldout_dir, test_fraction=0.1, min_cells=100):
    hd = Path(heldout_dir)
    ctx = jc.prepare(sr_dirs, kdir, ref_dir, min_cells)
    names, tx_len, unspliced = ctx["names"], ctx["tx_len"], ctx["unspliced"]
    t0 = time.time()
    _, test = jc.lr_libraries(ctx["lr_src"], names, 0.0, ctx["keep_g"]({g for s in ctx["lr_src"] for g in s[1]}),
                              AnchoredTruncationModel(tx_len, uniform=unspliced), test_fraction=float(test_fraction))
    print(f"held out: {len(test)} LR libraries ({time.time() - t0:.0f} s)", flush=True)
    genes = ctx["genes"]
    rows = []
    fractions = sorted({p.name.split("_")[1] for p in hd.glob("train_*") if p.is_dir()}, key=float)
    for frac in fractions:
        fits = {}
        for f in sorted(hd.glob(f"train_{frac}*/fits.pkl")):
            with open(f, "rb") as fh:
                for name, fit in pickle.load(fh)["fits"].items():
                    fits.setdefault(name, fit)            # joint is saved by several jobs: keep one
        if "LR" not in fits:
            continue
        keys = sorted(k for k in test if all(label_fn(f["present"])(k) in set(f["present"]) for f in fits.values()))
        lp = {}
        for name, fit in fits.items():
            t1 = time.time()
            lp[name] = fit_logprobs(fit, test, keys, genes)
            print(f"{frac} {name}: scored ({time.time() - t1:.0f} s)", flush=True)
        counts = next(iter(lp.values()))[2]
        for score, idx in [("all", 0), ("within_gene", 1)]:
            for l, k in enumerate(keys):
                ok = np.logical_and.reduce([np.isfinite(lp[n][idx][l]) for n in lp])
                n_all = counts[l][~np.isnan(lp["LR"][idx][l])].sum()
                for name in lp:
                    rows.append({"lr_train_fraction": float(frac), "fit": name, "score": score, "library": k,
                                 "unit": k.split("|")[0], "molecules": counts[l][ok].sum(), "molecules_scorable": n_all,
                                 "ll": counts[l][ok] @ lp[name][idx][l][ok]})
        del lp
    r = pd.DataFrame(rows)
    r.to_csv(hd / "rescore.tsv", sep="\t", index=False)
    lr = r[r.fit == "LR"].set_index(["lr_train_fraction", "score", "library"])["ll"]
    r["gain"] = r["ll"] - lr.reindex(pd.MultiIndex.from_frame(r[["lr_train_fraction", "score", "library"]])).to_numpy()
    s = r.groupby(["score", "fit", "lr_train_fraction"]).agg(gain=("gain", "sum"), molecules=("molecules", "sum"),
                                                              scorable=("molecules_scorable", "sum")).reset_index()
    s["gain_per_molecule"] = s["gain"] / s["molecules"]
    s["share_scored"] = s["molecules"] / s["scorable"]
    s.to_csv(hd / "rescore_summary.tsv", sep="\t", index=False)
    print(s.pivot_table(index="fit", columns=["score", "lr_train_fraction"], values="gain_per_molecule").round(4).to_string())
    print(s.pivot_table(index="score", columns="lr_train_fraction", values="share_scored", aggfunc="first").round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5], *(float(x) if i == 0 else int(x) for i, x in enumerate(sys.argv[5:7])))
