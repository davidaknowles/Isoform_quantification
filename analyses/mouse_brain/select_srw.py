"""
Choose the short-read weight (or other fit options) by held-out long-read likelihood: compares fit.py runs made
with the same lr_test (identical held-out molecules) from their heldout_lr.npz.

Per held-out library, ECs are kept when every run's joint fit and LR_bulk give them positive probability; the
score is the log-likelihood per held-out molecule of each run's joint fit minus LR_bulk's (all, and within gene:
reads of one-gene ECs conditioned on the gene). LR_bulk fits the bulk libraries directly, so it is the reference,
not a competitor: the runs are ranked against each other. Run labels are the directory names after "fit_".

Writes <out.tsv> (run x score x tissue) and prints the ranking.

Run: python analyses/mouse_brain/select_srw.py <out.tsv> <run_dir> [<run_dir> ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(out, *dirs):
    runs = {Path(d).name.removeprefix("fit_"): np.load(Path(d) / "heldout_lr.npz") for d in dirs}
    first = next(iter(runs.values()))
    libs = sorted({k.split("|")[0] for k in first.files})
    rows = []
    for lib in libs:
        counts = first[f"{lib}|counts"]
        for score in ("all", "within_gene"):
            arrs = {r: z[f"{lib}|joint|{score}"] for r, z in runs.items()}
            bulk = first[f"{lib}|LR_bulk|{score}"]
            ok = np.isfinite(bulk) & np.logical_and.reduce([np.isfinite(a) for a in arrs.values()])
            for r, a in arrs.items():
                rows.append({"library": lib, "run": r, "score": score, "molecules": counts[ok].sum(),
                             "gain": counts[ok] @ (a[ok] - bulk[ok])})
    r = pd.DataFrame(rows)
    s = r.groupby(["score", "run"]).agg(gain=("gain", "sum"), molecules=("molecules", "sum")).reset_index()
    s["gain_per_molecule"] = s.gain / s.molecules
    s.to_csv(out, sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(s.pivot_table(index="run", columns="score", values="gain_per_molecule").sort_values("all", ascending=False)
          .round(5).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
