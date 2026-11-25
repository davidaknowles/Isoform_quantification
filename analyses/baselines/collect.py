"""
Gather the SG-NEx baseline outputs into one TPM table per replicate,
{baselines}/quant/{cell}_rep{rep}_tpm.tsv.gz (transcripts of the reference cDNA FASTA as rows, one
column per fit, missing transcripts 0), the layout spikein_eval.py reads.

Columns ({LR} = cDNA or directcDNA):
  oarfish:{LR}           oarfish --model-coverage (published recommendation)
  oarfish+trunc:{LR}     oarfish --model-truncation
  lr-kallisto:{LR}       kallisto bus --long / quant-tcc --long TPM
  NanoCount:{LR}         NanoCount TPM
  IsoQuant:{LR}          IsoQuant unique_only transcript counts
  miniQuant:{LR}         miniQuant long reads alone
  miniQuant:SR+{LR}      miniQuant hybrid with the replicate's Illumina reads
  MPAQT:SR+{LR}          MPAQT with Illumina reads and lr-kallisto long-read counts
Long-read counts are molecule counts, so their TPM is count / total * 1e6 (no length division,
as for JOLI's molar fractions); tools that report TPM (lr-kallisto, NanoCount, miniQuant, MPAQT)
use their own.

Run: ~/venv/joli/bin/python analyses/baselines/collect.py [baselines_dir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SGNEX = Path("/gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex")
REPS = [("H9", 2), ("H9", 3), ("H9", 4), ("HEYA8", 1), ("HEYA8", 2), ("HEYA8", 3)]
LR = ["directcDNA", "cDNA"]


def fasta_names(path):
    with open(path) as f:
        return [line[1:].split()[0] for line in f if line.startswith(">")]


def tpm_from_counts(s):
    s = s.astype(float).groupby(level=0).sum()
    return 1e6 * s / s.sum()


def read_oarfish(p):
    df = pd.read_csv(p, sep="\t", index_col=0)
    return tpm_from_counts(df["num_reads"])


def read_lr_kallisto(d):
    # quant-tcc --matrix-to-files writes abundance tables per matrix row
    for name in ["abundance_1.tsv", "abundance.tsv"]:
        p = d / name
        if p.exists():
            df = pd.read_csv(p, sep="\t", index_col=0)
            return df["tpm"].astype(float)
    # fallback: matrix.abundance.tpm.mtx (one row, transcripts in transcripts.txt)
    p = d / "matrix.abundance.tpm.mtx"
    if p.exists():
        from scipy.io import mmread
        m = np.asarray(mmread(p).todense()).ravel()
        names = pd.read_csv(d / "transcripts.txt", header=None)[0]
        return pd.Series(m, index=names.values)
    return None


def read_nanocount(p):
    df = pd.read_csv(p, sep="\t", index_col=0)
    return df["tpm"].astype(float)


def read_isoquant(d, name):
    p = d / name / f"{name}.transcript_counts.tsv"
    if not p.exists():
        p = Path(str(p) + ".gz")
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t", index_col=0)
    return tpm_from_counts(df.iloc[:, 0])


def read_miniquant(p):
    df = pd.read_csv(p, sep="\t", index_col=0)
    return df["TPM"].astype(float)


def read_mpaqt(p):
    df = pd.read_csv(p, index_col=0)
    col = "tpm" if "tpm" in df.columns else df.columns[-1]
    return df[col].astype(float)


def main(bdir=SGNEX / "baselines"):
    bdir = Path(bdir)
    names = fasta_names(SGNEX / "ref" / "hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa")
    (bdir / "quant").mkdir(exist_ok=True)
    for cell, rep in REPS:
        cols = {}
        for lr in LR:
            name = f"{cell}_rep{rep}_{lr}"
            sources = {
                f"oarfish:{lr}": (read_oarfish, bdir / "oarfish" / f"{name}.cov.quant"),
                f"oarfish+trunc:{lr}": (read_oarfish, bdir / "oarfish" / f"{name}.trunc.quant"),
                f"lr-kallisto:{lr}": (read_lr_kallisto, bdir / "lr-kallisto" / name),
                f"NanoCount:{lr}": (read_nanocount, bdir / "nanocount" / f"{name}.tsv"),
                f"IsoQuant:{lr}": (lambda d, n=name: read_isoquant(d, n), bdir / "isoquant" / name),
                f"miniQuant:{lr}": (read_miniquant, bdir / "miniquant" / f"{name}.lr" / "abundance.tsv"),
                f"miniQuant:SR+{lr}": (read_miniquant, bdir / "miniquant" / f"{name}.hybrid" / "abundance.tsv"),
                f"MPAQT:SR+{lr}": (read_mpaqt, bdir / "mpaqt" / name / f"{name}.quant.csv"),
            }
            for col, (fn, path) in sources.items():
                if not path.exists():
                    continue
                s = fn(path)
                if s is None:
                    continue
                miss = len(set(s.index[s > 0]) - set(names))
                if miss:
                    print(f"  {col} {cell} rep{rep}: {miss} nonzero transcripts not in reference", flush=True)
                cols[col] = s.reindex(names).fillna(0.0)
        if not cols:
            print(f"{cell} rep{rep}: no outputs yet")
            continue
        df = pd.DataFrame(cols, index=names)
        df.to_csv(bdir / "quant" / f"{cell}_rep{rep}_tpm.tsv.gz", sep="\t", float_format="%.6g")
        print(f"{cell} rep{rep}: {', '.join(df.columns)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
