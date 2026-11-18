"""
Parse PBMC short reads alone, per cell type (the SR libraries and bias models of the full Parse + Kinnex run,
analyses/parse/joint_celltype.py, without long reads), for comparison with Smart-seq2 against the Kinnex
long-read cell-type profiles (fit_compare.py).

Writes <out_dir>/tpm.tsv.gz (columns SR|<cell type>) and fits.tsv.

Run: python analyses/pbmc_ss2/parse_sr.py <sr_series_dirs> <kinnex_dir> <ref_dir> <out_dir> [sr_fraction] [min_cells]
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parse"))
import joint_celltype as jc  # noqa: E402

from joli.joint import molar_to_tpm  # noqa: E402


def main(sr_dirs, kdir, ref_dir, out_dir, sr_fraction=1.0, min_cells=100):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = jc.prepare(sr_dirs, kdir, ref_dir, int(min_cells))
    names, tx_len, unspliced = ctx["names"], ctx["tx_len"], ctx["unspliced"]
    feat = jc.StackedFeatures([jc.ReadPositionModel(tx_len, jc.SR_READ_LEN, "3p", uniform=unspliced),
                               jc.ReadPositionModel(tx_len, jc.SR_READ_LEN, "relative", uniform=unspliced)])
    t0 = time.time()
    sr = jc.sr_libraries(ctx["sr_src"], names, float(sr_fraction), ctx["keep_g"]([g for s in ctx["sr_src"] for g in s[3]]), feat)
    print(f"{len(sr)} SR libraries ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    o = jc.fit_one(ctx, sr, True, False)
    res = o["res"]
    tpm = {f"SR|{u}": molar_to_tpm(res.theta[i]) for i, u in enumerate(o["present"])}
    pd.DataFrame(tpm, index=names).to_csv(out / "tpm.tsv.gz", sep="\t", float_format="%.6g")
    pd.DataFrame([{"fit": "SR", "loglik": res.loglik[-1], "n_iter": res.n_iter, "converged": res.converged,
                   "seconds": round(time.time() - t0)}]).to_csv(out / "fits.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:5], *(float(x) if i == 0 else int(x) for i, x in enumerate(sys.argv[5:7])))
