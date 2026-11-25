"""
Score mouse-brain fits against the single-cell long-read reference (truth.py): within-gene isoform fractions
per region x cell class, over genes with >= MIN_GENE_UMI truth UMIs on >= 2 reference transcripts and gene TPM
>= 1 in the estimate. Estimates (short reads alone do not identify isoforms and are not scored): LR_bulk (the region's
bulk tissue profile, the same for every class), joint base unit and joint consensus. Metrics: median gene TV
of isoform fractions and Pearson of isoform fractions over all scored transcripts.

Second table (evaluation_common.tsv), per region x class on genes scorable for every estimate of that class:
median gene TV for each estimate and two references from the truth itself, replicate M1 vs M2 (noise floor at
half depth) and the pooled truth of the region's other classes (a cell-type-agnostic long-read profile); and a
specificity margin, the mean over other classes c' of median TV(estimate of c', truth c) - median TV(estimate
of c, truth c) on genes scorable for both (positive: the class's own estimate fits its truth better; 0 for
LR_bulk by construction).

evaluation_diu_replicates.tsv repeats it against each truth replicate (M1, M2) separately, so a fit option can be
chosen on one replicate and scored on the other.

Third table (evaluation_diu.tsv), differential isoform usage: per region x class, genes whose truth isoform
fractions differ from the mean over the region's other classes by gene TV >= DIU_TV (truth >= DIU_UMI UMIs in the
class and in each other class used); Pearson over their transcripts of delta psi (class - mean of other classes)
between estimate and truth, and the share of genes where the estimate moves the truth's most-changed transcript
the same way.

Run: python analyses/mouse_brain/evaluate.py <fit_dir> <truth.tsv.gz> <ref_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_GENE_UMI = 20
DIU_UMI, DIU_TV = 100, 0.2


def within(est, truth, gene):
    """(median gene TV, psi Pearson, genes) between estimate and truth over eligible genes."""
    idx = truth.index.intersection(est.index)
    e, t, g = est.reindex(idx).fillna(0.0), truth.reindex(idx), gene.reindex(idx)
    gt, ge = t.groupby(g).transform("sum"), e.groupby(g).transform("sum")
    n = g.groupby(g).transform("size")
    ok = (gt >= MIN_GENE_UMI) & (ge >= 1.0) & (n >= 2)
    if not ok.any():
        return np.nan, np.nan, 0
    pe, pt = (e / ge)[ok], (t / gt)[ok]
    tv = (0.5 * (pe - pt).abs()).groupby(g[ok]).sum()
    return float(tv.median()), float(np.corrcoef(pe, pt)[0, 1]), int(tv.size)


def main(fit_dir, truth_path, ref_dir):
    fd = Path(fit_dir)
    tpm = pd.read_csv(fd / "tpm.tsv.gz", sep="\t", index_col=0)
    tpm = tpm[~tpm.index.str.endswith("-U")]
    truth = pd.read_csv(truth_path, sep="\t", index_col=0)
    t2g = pd.read_csv(Path(ref_dir) / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    gene = t2g.reindex(tpm.index)
    rows = []
    for col in [c for c in truth.columns if c.endswith("|all")]:
        region, cls = col.split("|")[:2]
        t = truth[col]
        cands = {}
        if f"LR_bulk|{region}" in tpm:
            cands["LR_bulk"] = tpm[f"LR_bulk|{region}"]
        for name, c in [("joint", f"joint|{region}|{cls}"), ("joint_consensus", f"joint|{region}|{cls}|consensus")]:
            if c in tpm:
                cands[name] = tpm[c]
        for name, est in cands.items():
            tv, r, ng = within(est, t, gene)
            rows.append({"region": region, "cell_class": cls, "estimate": name, "gene_tv_median": tv,
                         "psi_pearson": r, "genes": ng})
    r = pd.DataFrame(rows)
    r.to_csv(fd / "evaluation.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(r.pivot_table(index=["region", "cell_class"], columns="estimate", values="gene_tv_median").round(3).to_string())
    print(r.groupby("estimate")[["gene_tv_median", "psi_pearson", "genes"]].median().round(3).to_string())
    c = common(tpm, truth, gene)
    c.to_csv(fd / "evaluation_common.tsv", sep="\t", index=False)
    print(c.pivot_table(index=["region", "cell_class"], columns="estimate", values="tv").round(3).to_string())
    print(c.groupby("estimate")[["tv", "specificity", "genes"]].median().round(4).to_string())
    d = diu(tpm, truth, gene)
    reps = pd.concat([diu(tpm, truth, gene, r).assign(replicate=r) for r in ("M1", "M2")], ignore_index=True)
    reps.to_csv(fd / "evaluation_diu_replicates.tsv", sep="\t", index=False)
    d.to_csv(fd / "evaluation_diu.tsv", sep="\t", index=False)
    print(d.pivot_table(index=["region", "cell_class"], columns="estimate", values="dpsi_pearson").round(3).to_string())
    tot = d.groupby("estimate").apply(lambda x: pd.Series({"dpsi_pearson_median": x.dpsi_pearson.median(),
                                                           "sign_agree": (x.sign_agree * x.genes).sum() / x.genes.sum(),
                                                           "genes": x.genes.sum()}), include_groups=False)
    print(tot.round(3).to_string())


def diu(tpm, truth, gene, rep="all"):
    """Differential isoform usage of each class against the mean of the region's other classes."""
    idx = truth.index.intersection(tpm.index)
    tpm, truth, g = tpm.reindex(idx), truth.reindex(idx), gene.reindex(idx)
    n_tx = g.groupby(g).transform("size")
    rows = []
    for region in sorted({c.split("|")[0] for c in truth.columns if c.endswith("|all")}):
        classes = [c.split("|")[1] for c in truth.columns if c.endswith("|all") and c.startswith(region + "|")]
        est = {cls: estimates(tpm, region, cls) for cls in classes}
        tcol = {cls: truth[f"{region}|{cls}|{rep}"] for cls in classes if f"{region}|{cls}|{rep}" in truth}
        classes = list(tcol)
        tg = {cls: tcol[cls].groupby(g).transform("sum") for cls in classes}
        tpsi = {cls: tcol[cls] / tg[cls].replace(0, np.nan) for cls in classes}
        epsi = {cls: {n: e / e.groupby(g).transform("sum").replace(0, np.nan) for n, e in est[cls].items()}
                for cls in classes}
        for cls in classes:
            others = [o for o in classes if o != cls]
            deep = {o: tg[o] >= DIU_UMI for o in others}
            n_deep = sum(d.astype(int) for d in deep.values())
            ok = (tg[cls] >= DIU_UMI) & (n_deep >= 2) & (n_tx >= 2)
            mt = sum(tpsi[o].where(deep[o], 0.0).fillna(0.0) for o in others) / n_deep.replace(0, np.nan)
            dt = (tpsi[cls] - mt)[ok]
            gtv = (0.5 * dt.abs()).groupby(g[ok]).sum()
            sel = g[ok].isin(gtv.index[gtv >= DIU_TV])
            dt = dt[sel]
            gs = g[ok][sel]
            for n in sorted(est[cls]):
                avail = [i for i, o in enumerate(others) if n in est[o]]
                if not avail:
                    continue
                num = sum(epsi[others[i]][n].where(deep[others[i]], 0.0).fillna(0.0) for i in avail)
                cnt = sum(deep[others[i]].astype(float) for i in avail)
                de = (epsi[cls][n] - num / cnt.replace(0, np.nan)).reindex(dt.index)
                keep = de.notna()
                top = dt[keep].abs().groupby(gs[keep]).idxmax()
                agree = float((np.sign(de[top]) == np.sign(dt[top])).mean()) if len(top) else np.nan
                r = float(np.corrcoef(de[keep], dt[keep])[0, 1]) if keep.sum() > 2 and de[keep].std() > 0 else 0.0
                rows.append({"region": region, "cell_class": cls, "estimate": n, "dpsi_pearson": r,
                             "sign_agree": agree, "genes": int(gs[keep].nunique())})
    return pd.DataFrame(rows)


def gene_tv(e, t, g, ok):
    """Per-gene TV of within-gene fractions of e and t (Series on transcripts) over transcripts in ok."""
    e, t, g = e[ok], t[ok], g[ok]
    pe, pt = e / e.groupby(g).transform("sum"), t / t.groupby(g).transform("sum")
    return (0.5 * (pe - pt).abs()).groupby(g).sum()


def estimates(tpm, region, cls):
    """Estimate columns for one region x class (as in main)."""
    out = {}
    if f"LR_bulk|{region}" in tpm:
        out["LR_bulk"] = tpm[f"LR_bulk|{region}"]
    for name, col in [("joint", f"joint|{region}|{cls}"), ("joint_consensus", f"joint|{region}|{cls}|consensus")]:
        if col in tpm:
            out[name] = tpm[col]
    return out


def common(tpm, truth, gene):
    idx = truth.index.intersection(tpm.index)
    tpm, truth, g = tpm.reindex(idx), truth.reindex(idx), gene.reindex(idx)
    n_tx = g.groupby(g).transform("size")
    rows = []
    for region in sorted({c.split("|")[0] for c in truth.columns if c.endswith("|all")}):
        classes = [c.split("|")[1] for c in truth.columns if c.endswith("|all") and c.startswith(region + "|")]
        est = {cls: estimates(tpm, region, cls) for cls in classes}
        tgene = {cls: truth[f"{region}|{cls}|all"].groupby(g).transform("sum") for cls in classes}
        egene = {cls: {n: e.groupby(g).transform("sum") for n, e in est[cls].items()} for cls in classes}
        base = {cls: (tgene[cls] >= MIN_GENE_UMI) & (n_tx >= 2) for cls in classes}
        for cls in classes:
            names = sorted(est[cls])
            t = truth[f"{region}|{cls}|all"]
            ok = base[cls] & np.logical_and.reduce([egene[cls][n] >= 1.0 for n in names] or [base[cls]])
            m1, m2 = truth.get(f"{region}|{cls}|M1"), truth.get(f"{region}|{cls}|M2")
            refs = {}
            if m1 is not None and m2 is not None:
                ok_rep = ok & (m1.groupby(g).transform("sum") > 0) & (m2.groupby(g).transform("sum") > 0)
                refs["truth_M1_vs_M2"] = (m1, m2, ok_rep)
            others = [f"{region}|{o}|all" for o in classes if o != cls]
            if others:
                refs["truth_other_classes"] = (truth[others].sum(axis=1), t, ok)
            for n in names:
                tv = gene_tv(est[cls][n], t, g, ok)
                spec = []
                for o in classes:
                    if o == cls or n not in est[o]:
                        continue
                    ok2 = ok & (egene[o][n] >= 1.0)
                    spec.append(gene_tv(est[o][n], t, g, ok2).median() - gene_tv(est[cls][n], t, g, ok2).median())
                rows.append({"region": region, "cell_class": cls, "estimate": n, "tv": tv.median(),
                             "specificity": np.mean(spec) if spec else np.nan, "genes": tv.size})
            for n, (a, b, okr) in refs.items():
                tv = gene_tv(a, b, g, okr)
                rows.append({"region": region, "cell_class": cls, "estimate": n, "tv": tv.median(),
                             "specificity": np.nan, "genes": tv.size})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(*sys.argv[1:4])
