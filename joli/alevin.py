"""
alevin.py
=========
Pseudobulk JointEM libraries from alevin-fry transcript-level output
(`alevin-fry quant -m t2t.tsv --dump-eqclasses`): per-cell UMI counts over
equivalence classes of transcripts (here spliced transcripts and unspliced
gene targets), summed over groups of cells.

For Parse, each cell barcode is one RT primer half of a cell. Poly(dT)
priming gives one UMI per molecule near its 3' end, so its sampling rate
does not grow with length (b_t = 1); random-hexamer priming can tag a
molecule many times, so its rate is proportional to the effective length.
Both halves of a group observe the same molecules, so they are two libraries
of one JointEM unit.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from .joint import Library
from .load_tcc import TCCData


def _alevin_dir(path):
    path = Path(path)
    return path / "alevin" if (path / "alevin" / "quants_mat_rows.txt").exists() else path


def read_alevin_fry(quant_dir):
    """
    Returns:
        barcodes : list[str] (C,)
        features : list[str] (F,)        -- t2t feature names
        counts   : scipy.sparse.csr_matrix (C, E) -- UMIs per cell and EC
        ec_feat  : list[np.ndarray] (E,) -- feature indices of each EC
    """
    d = _alevin_dir(quant_dir)
    barcodes = (d / "quants_mat_rows.txt").read_text().split()
    features = (d / "quants_mat_cols.txt").read_text().split()
    counts = scipy.io.mmread(d / "geqc_counts.mtx").tocsr()
    with gzip.open(d / "gene_eqclass.txt.gz", "rt") as fh:
        n_feat = int(fh.readline())
        n_ec = int(fh.readline())
        ec_feat = [None] * n_ec
        for line in fh:
            f = np.array(line.split(), dtype=np.int64)
            ec_feat[f[-1]] = f[:-1]
    if n_feat != len(features) or counts.shape != (len(barcodes), n_ec):
        raise ValueError(f"inconsistent alevin-fry output in {d}")
    return barcodes, features, counts, ec_feat


def pseudobulk_counts(quant_dirs, cells, group_cols=("subject", "cell_type", "primer")):
    """
    Sum UMI counts over cells of each group, pooling ECs across quant
    directories by their transcript sets.

    Args:
        quant_dirs : dict {library: alevin-fry quant dir}; cells.library selects the dir.
        cells      : DataFrame with columns barcode, library and group_cols.
    Returns:
        features : list[str]
        ec_keys  : list[tuple[str]] (E,) -- transcript names of each pooled EC
        groups   : DataFrame (G, group_cols + n_cells)
        counts   : np.ndarray (G, E)
    """
    group_cols = list(group_cols)
    gkey = cells[group_cols].astype(str).agg("|".join, axis=1)
    group_names = sorted(gkey.unique())
    gidx = pd.Series(np.arange(len(group_names)), index=group_names)
    ec_index, rows = {}, []
    features = None
    for lib, qdir in quant_dirs.items():
        barcodes, feats, counts, ec_feat = read_alevin_fry(qdir)
        if features is None:
            features = feats
        elif feats != features:
            raise ValueError(f"{lib}: feature list differs from the first library")
        sel = cells[cells["library"] == lib]
        pos = pd.Series(np.arange(len(barcodes)), index=barcodes)
        keep = sel["barcode"].isin(pos.index).to_numpy()
        r = pos[sel["barcode"][keep]].to_numpy()
        g = gidx[gkey[sel.index[keep]]].to_numpy()
        A = sp.csr_matrix((np.ones(len(r)), (g, r)), shape=(len(group_names), len(barcodes)))
        summed = (A @ counts).tocoo()
        local = np.array([ec_index.setdefault(tuple(ec_feat[e]), len(ec_index)) for e in range(len(ec_feat))])
        rows.append((summed.row, local[summed.col], summed.data))
    G, E = len(group_names), len(ec_index)
    total = np.zeros((G, E))
    for gr, ec, v in rows:
        np.add.at(total, (gr, ec), v)
    ec_keys = [None] * E
    for key, e in ec_index.items():
        ec_keys[e] = tuple(features[i] for i in key)
    groups = pd.DataFrame([n.split("|") for n in group_names], columns=group_cols)
    groups["n_cells"] = gkey.value_counts().reindex(group_names).to_numpy()
    return features, ec_keys, groups, total


def library_from_pseudobulk(name, ec_keys, counts, transcript_names, b, platform="short") -> Library:
    """One group's counts (E,) as a Library over transcript_names with rates b (T,)."""
    pos = pd.Series(np.arange(len(transcript_names)), index=transcript_names)
    keep = np.flatnonzero(counts > 0)
    ec_tx = [pos[list(ec_keys[e])].to_numpy() for e in keep]
    tcc = TCCData(ec_counts=counts[keep].astype(np.float64), ec_transcripts=ec_tx,
                  transcript_names=list(transcript_names), total_reads=int(counts[keep].sum()),
                  n_targets=len(transcript_names), sample_dir=name)
    return Library(name=name, tcc=tcc, b=np.asarray(b, dtype=np.float64), platform=platform)


def primer_rates(primer, eff_len):
    """Sampling rate per transcript for a Parse RT primer: 1 for poly(dT), effective length for random hexamer."""
    if primer == "poly(dT)":
        return np.ones_like(eff_len, dtype=np.float64)
    if primer == "random hexamer":
        return np.asarray(eff_len, dtype=np.float64)
    raise ValueError(f"unknown primer {primer}")
