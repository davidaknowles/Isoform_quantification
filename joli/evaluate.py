"""
evaluate.py
===========
Compare estimated transcript abundances to a ground truth.

Transcript-level metrics use TPM on the union of transcripts with nonzero
estimate or truth (true negatives excluded). Within-gene metrics compare
isoform fractions psi[t] = tpm[t] / sum_{t' in gene(t)} tpm[t'] over genes
with at least two transcripts and nonzero true expression, so that sharing
across samples is judged on isoform usage rather than gene expression.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def read_tpm(path, id_col=None, tpm_col="tpm", sep=None) -> pd.Series:
    """
    Read a TPM table as a Series indexed by transcript id.

    Handles kallisto/JOLI abundance.tsv (target_id or transcript_id), the
    simulation ground truths (transcript_name, comma or tab separated), and
    ids of the form "NM_000001.1|..." (truncated at the first "|").
    """
    df = pd.read_csv(path, sep=sep, engine="python")
    if id_col is None:
        id_col = next(c for c in ("target_id", "transcript_id", "transcript_name") if c in df.columns)
    ids = df[id_col].astype(str).str.split("|").str[0]
    s = pd.Series(df[tpm_col].to_numpy(dtype=float), index=ids)
    return s.groupby(level=0).sum()


def read_t2g(path) -> pd.Series:
    """Transcript -> gene Series from a headerless tab-separated t2g file (first two columns)."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], dtype=str)
    return pd.Series(df[1].to_numpy(), index=df[0].to_numpy())


def transcript_metrics(est: pd.Series, truth: pd.Series) -> dict:
    """Detection counts and correlations on the union of nonzero transcripts."""
    m = pd.concat([est.rename("est"), truth.rename("truth")], axis=1).fillna(0.0)
    e, t = m["est"].to_numpy(), m["truth"].to_numpy()
    pos_e, pos_t = e > 0, t > 0
    u = pos_e | pos_t
    return {
        "n_union": int(u.sum()),
        "tp": int((pos_e & pos_t).sum()),
        "fp": int((pos_e & ~pos_t).sum()),
        "fn": int((~pos_e & pos_t).sum()),
        "fp_tpm": float(e[pos_e & ~pos_t].sum()),
        "spearman": float(spearmanr(e[u], t[u])[0]),
        "pearson_log1p": float(pearsonr(np.log1p(e[u]), np.log1p(t[u]))[0]),
        "mard": float(np.mean(np.abs(e[u] - t[u]) / (e[u] + t[u]))),
    }


def isoform_fractions(tpm: pd.Series, t2g: pd.Series) -> pd.DataFrame:
    """Per-transcript gene id, gene TPM and isoform fraction psi."""
    df = pd.DataFrame({"tpm": tpm})
    df["gene"] = t2g.reindex(df.index).to_numpy()
    df = df.dropna(subset=["gene"])
    gene_tpm = df.groupby("gene")["tpm"].transform("sum")
    df["gene_tpm"] = gene_tpm
    df["psi"] = np.where(gene_tpm > 0, df["tpm"] / gene_tpm.where(gene_tpm > 0, 1.0), np.nan)
    return df


def fraction_metrics(est: pd.Series, truth: pd.Series, t2g: pd.Series, min_gene_tpm: float = 1.0) -> dict:
    """
    Within-gene isoform-fraction accuracy.

    Restricted to genes with >= 2 annotated transcripts and true gene TPM >=
    min_gene_tpm. Transcripts of genes with zero estimated expression get the
    uniform fraction 1/n_gene.
    """
    idx = t2g.index
    t_df = isoform_fractions(truth.reindex(idx).fillna(0.0), t2g)
    e_df = isoform_fractions(est.reindex(idx).fillna(0.0), t2g)
    n_tx = t_df.groupby("gene")["tpm"].transform("size")
    keep = (n_tx >= 2) & (t_df["gene_tpm"] >= min_gene_tpm)
    psi_t = t_df.loc[keep, "psi"].to_numpy()
    psi_e = e_df.loc[keep.index[keep], "psi"].to_numpy()
    psi_e = np.where(np.isnan(psi_e), 1.0 / n_tx[keep].to_numpy(), psi_e)
    genes = t_df.loc[keep, "gene"].to_numpy()
    abs_err = pd.Series(np.abs(psi_e - psi_t)).groupby(genes).sum() / 2.0   # total variation per gene
    major_t = pd.Series(psi_t, index=t_df.index[keep]).groupby(genes).idxmax()
    major_e = pd.Series(psi_e, index=t_df.index[keep]).groupby(genes).idxmax()
    return {
        "n_genes": int(len(abs_err)),
        "psi_pearson": float(pearsonr(psi_e, psi_t)[0]),
        "psi_mean_abs_err": float(np.mean(np.abs(psi_e - psi_t))),
        "gene_tv_median": float(abs_err.median()),
        "major_isoform_acc": float((major_t == major_e.reindex(major_t.index)).mean()),
    }
