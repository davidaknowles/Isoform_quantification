"""
Summarise held-out LR likelihoods across training fractions (heldout_lr.py outputs).

Per training fraction: log-likelihood per held-out molecule of LR alone and joint,
their difference (joint - LR; > 0 means short reads improve prediction of unseen
long reads), and the share of held-out molecules on transcripts with zero fitted
abundance. Per cell type: the same difference, against the cell type's number of
held-out molecules. Libraries are scored only where both fits cover the cell type.

Writes {heldout_dir}/summary.tsv, summary_by_unit.tsv and heldout_gain.png.

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
    eps_cols = [c for c in h.columns if c.startswith("ll_eps")]
    long = h.melt(id_vars=["fit", "library", "unit", "kit", "molecules", "zero_prob_molecules", "lr_train_fraction"],
                  value_vars=eps_cols, var_name="eps", value_name="ll")
    long["eps"] = long["eps"].str.replace("ll_eps", "")
    w = long.pivot_table(index=["lr_train_fraction", "eps", "library", "unit", "kit", "molecules"], columns="fit",
                         values="ll").dropna().reset_index()
    z = h.pivot_table(index=["lr_train_fraction", "library"], columns="fit", values="zero_prob_molecules").reset_index()
    w = w.merge(z.rename(columns={"LR": "zero_LR", "joint": "zero_joint"}), on=["lr_train_fraction", "library"])

    def agg(g):
        n = g["molecules"].sum()
        return pd.Series({"molecules": n, "ll_LR": g["LR"].sum() / n, "ll_joint": g["joint"].sum() / n,
                          "gain_per_molecule": (g["joint"] - g["LR"]).sum() / n,
                          "zero_share_LR": g["zero_LR"].sum() / n, "zero_share_joint": g["zero_joint"].sum() / n,
                          "libraries": len(g)})

    s = w.groupby(["lr_train_fraction", "eps"]).apply(agg, include_groups=False).reset_index()
    u = w.groupby(["lr_train_fraction", "eps", "unit"]).apply(agg, include_groups=False).reset_index()
    s.to_csv(d / "summary.tsv", sep="\t", index=False)
    u.to_csv(d / "summary_by_unit.tsv", sep="\t", index=False)
    print(s.round(4).to_string(index=False))

    eps0 = sorted(s["eps"].unique(), key=float)[-1]               # larger smoothing
    p = (ggplot(u[u["eps"] == eps0], aes("lr_train_fraction", "gain_per_molecule"))
         + geom_hline(yintercept=0, linetype="dashed")
         + geom_line(aes(group="unit"), alpha=0.4)
         + geom_point(aes(size="molecules"), alpha=0.6)
         + geom_line(data=s[s["eps"] == eps0], color="red", size=1.2)
         + scale_x_log10()
         + labs(x="LR training fraction of molecules", y="held-out LR log-lik per molecule, joint - LR",
                title=f"Short reads and prediction of held-out long reads (eps {eps0}); red: all cell types")
         + theme_bw())
    p.save(d / "heldout_gain.png", width=8, height=5, dpi=150, verbose=False)


if __name__ == "__main__":
    main(sys.argv[1])
