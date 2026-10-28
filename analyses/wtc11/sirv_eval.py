"""
Score long-read fits on SIRV Set 4 spike-ins. The 69 SIRV isoforms (7 genes)
and the 15 long SIRVs are each at 0.6 fmol, so within a SIRV gene the true
isoform fraction is 1/n, and all 84 transcripts have equal molar abundance.

Metrics per fit and biosample: reads-weighted share of the spike-in mass
(sanity), gene TV of psi against 1/n (median and mean over the 7 genes),
SD of log molar abundance across the 69 isoforms and across all 84
(0 = perfect), and the ratio of mean long-SIRV to mean SIRV-isoform abundance
(1 = no length bias).

Run: python analyses/wtc11/sirv_eval.py <quant_dir> <sirv_t2g.tsv> [out.tsv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(quant_dir, t2g_path, out_path=None):
    t2g = pd.read_csv(t2g_path, sep="\t", header=None, index_col=0)[1]
    sirv = t2g[t2g.str.match(r"^SIRV[1-7]$")]
    long_ = t2g[t2g.str.match(r"^SIRV\d{4,5}$")]
    rows = []
    for f in sorted(Path(quant_dir).glob("*_tpm.tsv.gz")):
        df = pd.read_csv(f, sep="\t", index_col=0)
        for fit in df.columns:
            x = df[fit]
            iso = x.reindex(sirv.index).fillna(0.0)
            lng = x.reindex(long_.index).dropna()
            psi = iso / iso.groupby(sirv).transform("sum")
            n = sirv.groupby(sirv).transform("size")
            tv = (psi - 1.0 / n).abs().groupby(sirv).sum() / 2.0
            floor = 1e-3 * iso[iso > 0].median()
            allv = pd.concat([iso, lng])
            rows.append({
                "biosample": f.name.split("_")[0], "fit": fit,
                "sirv_tpm": float(iso.sum()), "n_zero": int((iso == 0).sum()),
                "sirv_gene_tv_median": float(tv.median()), "sirv_gene_tv_mean": float(tv.mean()),
                "sd_log_iso": float(np.log(iso + floor).std()),
                "sd_log_all": float(np.log(allv + floor).std()),
                "long_over_iso": float(lng.mean() / iso.mean()) if len(lng) else np.nan,
            })
    out = pd.DataFrame(rows)
    print(out.groupby("fit", sort=False).mean(numeric_only=True).round(3).to_string())
    if out_path:
        out.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
