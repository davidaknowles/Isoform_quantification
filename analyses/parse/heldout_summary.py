"""
Summarise held-out LR likelihoods across training fractions (heldout_lr.py outputs).

Per training fraction: log-likelihood per held-out molecule of LR alone and joint,
their difference (joint - LR; > 0 means short reads improve prediction of unseen
long reads), and the share of held-out molecules on transcripts with zero fitted
abundance. Per cell type: the same difference, against the cell type's number of
held-out molecules. Libraries are scored only where both fits cover the cell type.

Scores: all (log-likelihood of the read) and within_gene (given its gene, single-gene
ECs only). Writes {heldout_dir}/summary.tsv, summary_by_unit.tsv and heldout_gain.png.

Run: python analyses/parse/heldout_summary.py <heldout_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (aes, facet_wrap, geom_hline, geom_line, geom_point, ggplot, labs, scale_x_log10,
                      theme_bw)


def main(d):
    d = Path(d)
    h = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(d.glob("train_*/heldout.tsv"))], ignore_index=True)
    # fits repeated across output directories (e.g. joint as a warm start of pooled fits): keep one
    h = h.drop_duplicates(["fit", "library", "lr_train_fraction"])
    eps_cols = [c for c in h.columns if c.startswith("ll_eps")]
    long = h.melt(id_vars=["fit", "library", "unit", "kit", "molecules", "zero_prob_molecules", "lr_train_fraction"],
                  value_vars=eps_cols, var_name="eps", value_name="ll")
    long["eps"] = long["eps"].str.replace("ll_eps", "")
    long["score"] = "all"
    if "ll_within_eps1e-06" in h.columns:
        # within-gene score: molecules counted are those used (single-gene ECs) under each eps
        parts = []
        for c in [c for c in h.columns if c.startswith("ll_within_eps")]:
            e = c.replace("ll_within_eps", "")
            x = h[["fit", "library", "unit", "kit", "lr_train_fraction", "within_zero_prob_molecules", c,
                   f"within_molecules_eps{e}"]].rename(columns={c: "ll", f"within_molecules_eps{e}": "molecules",
                                                               "within_zero_prob_molecules": "zero_prob_molecules"})
            x["eps"], x["score"] = e, "within_gene"
            parts.append(x)
        long = pd.concat([long] + parts, ignore_index=True)
    # molecules per library and score from the LR fit (both fits score the same molecules)
    nm = long[long["fit"] == "LR"].groupby(["lr_train_fraction", "eps", "score", "library"])["molecules"].first()
    long = long.drop(columns="molecules").join(nm, on=["lr_train_fraction", "eps", "score", "library"])
    w = long.pivot_table(index=["lr_train_fraction", "eps", "score", "library", "unit", "kit", "molecules"],
                         columns="fit", values="ll").reset_index()
    z = long.pivot_table(index=["lr_train_fraction", "eps", "score", "library"], columns="fit",
                         values="zero_prob_molecules").reset_index()
    fits = [f for f in long["fit"].unique() if f != "LR"]
    w = w.merge(z.rename(columns={f: f"zero_{f}" for f in ["LR"] + fits}),
                on=["lr_train_fraction", "eps", "score", "library"])
    # long format over the fits compared with LR alone
    w = pd.concat([w.assign(fit=f, other=w[f], zero_other=w[f"zero_{f}"]).dropna(subset=["LR", "other"])
                   for f in fits if f in w.columns], ignore_index=True)

    def agg(g):
        n = g["molecules"].sum()
        return pd.Series({"molecules": n, "ll_LR": g["LR"].sum() / n, "ll_fit": g["other"].sum() / n,
                          "gain_per_molecule": (g["other"] - g["LR"]).sum() / n,
                          "zero_share_LR": g["zero_LR"].sum() / n, "zero_share_fit": g["zero_other"].sum() / n,
                          "libraries": len(g)})

    s = w.groupby(["score", "fit", "lr_train_fraction", "eps"]).apply(agg, include_groups=False).reset_index()
    u = w.groupby(["score", "fit", "lr_train_fraction", "eps", "unit"]).apply(agg, include_groups=False).reset_index()
    s.to_csv(d / "summary.tsv", sep="\t", index=False)
    u.to_csv(d / "summary_by_unit.tsv", sep="\t", index=False)
    print(s.round(4).to_string(index=False))

    eps0 = sorted(s["eps"].unique(), key=float)[-1]               # larger smoothing
    p = (ggplot(u[u["eps"] == eps0], aes("lr_train_fraction", "gain_per_molecule"))
         + facet_wrap("~score + fit", scales="free_y")
         + geom_hline(yintercept=0, linetype="dashed")
         + geom_line(aes(group="unit"), alpha=0.4)
         + geom_point(aes(size="molecules"), alpha=0.6)
         + geom_line(data=s[s["eps"] == eps0], color="red", size=1.2)
         + scale_x_log10()
         + labs(x="LR training fraction of molecules", y="held-out LR log-lik per molecule, fit - LR",
                title=f"Short reads and prediction of held-out long reads (eps {eps0}); red: all cell types")
         + theme_bw())
    p.save(d / "heldout_gain.png", width=8, height=5, dpi=150, verbose=False)


if __name__ == "__main__":
    main(sys.argv[1])
