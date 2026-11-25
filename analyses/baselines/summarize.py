"""
Score the SG-NEx baselines and the JOLI fits on the spike-ins (analyses/sgnex/spikein_eval.py) and
write {baselines}/summary.tsv: mean over replicates of gene_tv_median, sd_log_ratio and log_pearson
per fit and mix, with the number of replicates scored. Per-replicate scores go to
{baselines}/scores_baselines.tsv and scores_joli.tsv.

JOLI comparators (from quant_anch): directcDNA+anch, cDNA+anch, SRfrag+directcDNA+anch,
SRfrag+cDNA+anch, SRfrag+allLR+anch; salmon on Illumina (SR:nobias, SR:bias) is kept for reference.

Run: ~/venv/joli/bin/python analyses/baselines/summarize.py [sgnex_dir]
"""

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sgnex"))
import spikein_eval  # noqa: E402

JOLI = ["directcDNA+anch", "cDNA+anch", "SRfrag+directcDNA+anch", "SRfrag+cDNA+anch",
        "SRfrag+allLR+anch", "SR:nobias", "SR:bias"]
MIX_ORDER = ["sequinA_v2", "sirv4", "sirv_E2"]


def scores(quant_dir, truth, out):
    with contextlib.redirect_stdout(io.StringIO()):
        spikein_eval.main(quant_dir, truth, out)
    return pd.read_csv(out, sep="\t")


def main(sgnex=Path("/gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex")):
    sgnex = Path(sgnex)
    bdir = sgnex / "baselines"
    truth = sgnex / "ref" / "spikein_truth.tsv"
    base = scores(bdir / "quant", truth, bdir / "scores_baselines.tsv")
    joli = scores(sgnex / "quant_anch", truth, bdir / "scores_joli.tsv")
    joli = joli[joli["fit"].isin(JOLI)]
    joli["fit"] = "JOLI " + joli["fit"]
    joli.loc[joli["fit"].str.startswith("JOLI SR:"), "fit"] = joli["fit"].str.replace("JOLI SR:", "salmon:SR ")
    df = pd.concat([joli, base])
    summ = (df.groupby(["mix", "fit"], sort=False)
            .agg(gene_tv_median=("gene_tv_median", "mean"), sd_log_ratio=("sd_log_ratio", "mean"),
                 log_pearson=("log_pearson", "mean"), n_rep=("rep", "size")).reset_index())
    summ["mix"] = pd.Categorical(summ["mix"], MIX_ORDER, ordered=True)
    summ = summ.sort_values(["mix", "gene_tv_median"])
    summ.to_csv(bdir / "summary.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    for mix, g in summ.groupby("mix", observed=True):
        print(f"\n== {mix}")
        print(g.drop(columns="mix").round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:2])
