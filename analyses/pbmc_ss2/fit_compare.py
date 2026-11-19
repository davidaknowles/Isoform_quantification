"""
Smart-seq2 PBMC short reads (make_groups.py, build_libraries.py) against the 10x Kinnex single-cell long reads.

Fit: SR_only, one unit per cell type x donor group (FragmentModel per library), as in the mouse-brain pilot.
Reference: the long-read-only cell-type fit of the full PBMC run (column "LR|<cell type>" of its tpm.tsv.gz;
Kinnex 10x 3'/5', read-end model), at the cell types both have. Within-gene isoform fractions are compared
on genes with >= 2 spliced isoforms and gene TPM >= MIN_TPM in both:
  tv           median gene TV between the SS2 estimate (mean over donors) and the LR profile;
  donor_tv     median gene TV between the two SS2 donors (cell types with both; SR noise and donor effects);
  specificity  mean over other types c' of TV(SS2 of c', LR of c) - TV(SS2 of c, LR of c) (positive: the
               type's own SS2 estimate fits its LR profile better);
  DIU          genes where the LR profile of c differs from the mean of the other types by gene TV >= DIU_TV
               (gene TPM >= DIU_TPM in c and each other type used): Pearson over their transcripts of
               delta psi (c - mean of other types), SS2 against LR, and the share of genes where SS2 moves
               the LR's most-changed transcript the same way.

With a Parse SR-only fit (parse_sr.py, columns SR|<cell type>) the same scores are reported for it, on the
cell types both short-read sets have.

Writes <out_dir>/tpm.tsv.gz, fits.tsv and compare.tsv (estimate x cell type).

Run: python analyses/pbmc_ss2/fit_compare.py <libraries.pkl> <lr_tpm.tsv.gz> <t2g.tsv> <out_dir> [parse_tpm.tsv.gz]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import FragmentModel, fit_bias

RUN_KW = dict(max_iter=20000, ll_rel_tol=1e-10)
MIN_TPM = 5.0
DIU_TPM, DIU_TV = 20.0, 0.2


def fit_sr(z):
    keys = sorted(z["sr"])
    fms = {}
    for i, g in enumerate(keys):
        fm = FragmentModel(z["tx_len"])
        fm.pf = z["sr"][g][1].copy()
        fms[i] = fm
    t0 = time.time()
    em = JointEM([z["sr"][g][0] for g in keys], units=list(range(len(keys))))
    res = fit_bias(em, frag=fms, n_outer=8, run_kw=RUN_KW, verbose=True)
    tpm = pd.DataFrame({f"SS2|{g}": molar_to_tpm(res.theta[i]) for i, g in enumerate(keys)}, index=z["names"])
    return tpm, {"fit": "SR_only", "loglik": res.loglik[-1], "n_iter": res.n_iter, "converged": res.converged,
                 "seconds": round(time.time() - t0)}


def psi(x, g):
    s = x.groupby(g).transform("sum")
    return x / s.where(s > 0)


def gene_tv(a, b, g, ok):
    pa, pb = psi(a[ok], g[ok]), psi(b[ok], g[ok])
    return (0.5 * (pa - pb).abs()).groupby(g[ok]).sum()


def compare(ss2, lr, gene, donors):
    rows, diu = [], []
    types = sorted(set(ss2) & set(lr))
    n_tx = gene.groupby(gene).transform("size")
    gs = {c: ss2[c].groupby(gene).transform("sum") for c in types}
    gl = {c: lr[c].groupby(gene).transform("sum") for c in types}
    for c in types:
        ok = (gs[c] >= MIN_TPM) & (gl[c] >= MIN_TPM) & (n_tx >= 2)
        tv = gene_tv(ss2[c], lr[c], gene, ok)
        spec = [gene_tv(ss2[o], lr[c], gene, ok & (gs[o] >= MIN_TPM)).median()
                - gene_tv(ss2[c], lr[c], gene, ok & (gs[o] >= MIN_TPM)).median() for o in types if o != c]
        row = {"cell_type": c, "tv": tv.median(), "specificity": np.mean(spec), "genes": tv.size}
        d = donors.get(c)
        if d is not None:
            okd = ok & (d[0].groupby(gene).transform("sum") >= MIN_TPM) & (d[1].groupby(gene).transform("sum") >= MIN_TPM)
            row["donor_tv"] = gene_tv(d[0], d[1], gene, okd).median()
        # differential isoform usage against the mean of the other types
        others = [o for o in types if o != c]
        deep = {o: gl[o] >= DIU_TPM for o in others}
        n_deep = sum(v.astype(int) for v in deep.values())
        okd = (gl[c] >= DIU_TPM) & (gs[c] >= MIN_TPM) & (n_deep >= 2) & (n_tx >= 2)
        ml = sum(psi(lr[o], gene).where(deep[o], 0.0).fillna(0.0) for o in others) / n_deep.where(n_deep > 0)
        ms = sum(psi(ss2[o], gene).where(deep[o], 0.0).fillna(0.0) for o in others) / n_deep.where(n_deep > 0)
        dl = (psi(lr[c], gene) - ml)[okd]
        gtv = (0.5 * dl.abs()).groupby(gene[okd]).sum()
        sel = gene[okd].isin(gtv.index[gtv >= DIU_TV])
        dl = dl[sel]
        de = (psi(ss2[c], gene) - ms)[okd][sel]
        keep = de.notna() & dl.notna()
        top = dl[keep].abs().groupby(gene[okd][sel][keep]).idxmax()
        row["diu_genes"] = int(len(top))
        row["diu_pearson"] = float(np.corrcoef(de[keep], dl[keep])[0, 1]) if keep.sum() > 2 else np.nan
        row["diu_sign_agree"] = float((np.sign(de[top]) == np.sign(dl[top])).mean()) if len(top) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(pkl, lr_tpm, t2g_path, out_dir, parse_tpm=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "tpm.tsv.gz").exists():
        tpm = pd.read_csv(out / "tpm.tsv.gz", sep="\t", index_col=0)
    else:
        z = pd.read_pickle(pkl)
        tpm, summ = fit_sr(z)
        pd.DataFrame([summ]).to_csv(out / "fits.tsv", sep="\t", index=False)
        tpm.to_csv(out / "tpm.tsv.gz", sep="\t", float_format="%.6g")
        print(summ, flush=True)
    tpm = tpm[~tpm.index.str.endswith("-U")]
    lr = pd.read_csv(lr_tpm, sep="\t", index_col=0, usecols=lambda c: c == "Unnamed: 0" or c.startswith("LR|"))
    lr.columns = lr.columns.str.split("|").str[1]
    lr = lr.reindex(tpm.index).fillna(0.0)
    gene = pd.read_csv(t2g_path, sep="\t", header=None, index_col=0)[1].reindex(tpm.index)
    ok = gene.notna()
    tpm, lr, gene = tpm[ok], lr[ok], gene[ok]
    groups = pd.Series(tpm.columns).str.split("|", expand=True)                  # SS2, cell type, donor
    ss2 = pd.DataFrame({c: tpm[[k for k in tpm.columns if k.split("|")[1] == c]].mean(axis=1)
                        for c in groups[1].unique()})
    donors = {c: [tpm[f"SS2|{c}|{d}"] for d in ("PBMC1", "PBMC2")] for c in groups[1].unique()
                           if all(f"SS2|{c}|{d}" in tpm for d in ("PBMC1", "PBMC2"))}
    ests = {"SS2": ss2}
    if parse_tpm is not None:
        p = pd.read_csv(parse_tpm, sep="\t", index_col=0, usecols=lambda c: c == "Unnamed: 0" or c.startswith("SR|"))
        p.columns = p.columns.str.split("|").str[1]
        p = p.reindex(tpm.index).fillna(0.0)
        common = sorted(set(p.columns) & set(ss2.columns))
        ests = {"SS2": ss2[common], "Parse": p[common]}
    r = pd.concat([compare(e, lr, gene, donors if n == "SS2" else {}).assign(estimate=n) for n, e in ests.items()])
    r.to_csv(out / "compare.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    print(r.drop(columns="cell_type").groupby("estimate").median().round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:6])
