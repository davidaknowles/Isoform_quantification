"""
paf.py
======
Long-read equivalence classes from minimap2 transcriptome alignments (PAF).

Reads are aligned to transcript sequences with secondary alignments kept
(``minimap2 -c -x map-hifi -N 100``). For each read the alignments whose
score is at least ``score_frac`` times the read's best are retained. The
default keeps only best-score ties: on the LRGASP-style simulation a 0.95
threshold (oarfish's default) admits isoforms that differ by whole exons
and is far less accurate (LR-only log-Pearson 0.74 vs 0.96). Unlike kallisto's TCC output, the alignment table
keeps where on each transcript the read lands, which a truncation or
coverage model needs.

Per-alignment weights w_rt (e.g. from a truncation model) are collapsed into
rich equivalence classes as salmon does: reads with the same transcript set
form one class and c_et is the mean over its reads of w_rt / sum_t' w_rt'.
"""

import numpy as np
import pandas as pd

from .joint import Library
from .load_tcc import TCCData

PAF_COLS = ["read", "qlen", "qstart", "qend", "strand", "tx", "tlen", "tstart", "tend",
            "nmatch", "alen", "mapq"]


def read_fasta_lengths(path):
    """Transcript names (first header word) and sequence lengths of a fasta, in file order."""
    names, lens, n = [], [], 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if names:
                    lens.append(n)
                names.append(line[1:].split()[0])
                n = 0
            else:
                n += len(line.strip())
    lens.append(n)
    return names, np.array(lens)


def read_paf(path, chunksize=2_000_000):
    """
    Read a minimap2 PAF (with -c, so AS:i is present) into a DataFrame with
    columns PAF_COLS + ["AS"]. Read and transcript names are categoricals.
    """
    parts = []
    for chunk in pd.read_csv(path, sep="\t", header=None, usecols=range(15), chunksize=chunksize,
                             names=PAF_COLS + ["t12", "t13", "t14"], dtype={"strand": "category"},
                             engine="c", quoting=3):
        tags = chunk[["t12", "t13", "t14"]].astype(str)
        AS = np.full(len(chunk), np.nan)
        for c in tags:
            m = tags[c].str.startswith("AS:i:")
            AS[m.to_numpy()] = tags[c][m].str[5:].astype(float).to_numpy()
        chunk = chunk[PAF_COLS].copy()
        chunk["AS"] = AS
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df["read"] = df["read"].astype("category")
    df["tx"] = df["tx"].astype("category")
    return df


def filter_alignments(aln, score_frac=1.0, strand=None, min_read_frac=0.0):
    """
    Keep, per read, alignments with AS >= score_frac * best AS, one per
    (read, transcript) pair (highest AS). Adds column AS_gap = best AS - AS.

    Args:
        strand        : "+" | "-" | None -- required alignment strand (e.g. "+" for direct RNA).
        min_read_frac : float -- minimum aligned fraction of the read (qend - qstart) / qlen.
    """
    a = aln
    if strand is not None:
        a = a[a["strand"] == strand]
    if min_read_frac > 0:
        a = a[(a["qend"] - a["qstart"]) >= min_read_frac * a["qlen"]]
    a = a.sort_values(["read", "tx", "AS"], ascending=[True, True, False])
    a = a.drop_duplicates(["read", "tx"], keep="first")
    best = a.groupby("read", observed=True)["AS"].transform("max")
    a = a[a["AS"] >= score_frac * best].copy()
    a["AS_gap"] = (best[a.index] - a["AS"]).to_numpy()
    return a.reset_index(drop=True)


def score_weight(aln, beta=1.0):
    """
    Per-alignment weight exp(-beta * (best AS of the read - AS)), salmon's
    --scoreExp form. As rich-EC weights these are averaged over reads with the
    same transcript set, which mixes tied reads with reads that have large gaps
    and loses most of the information (worse than strict filtering in simulation).
    """
    return np.exp(-beta * aln["AS_gap"].to_numpy(dtype=np.float64))


def library_from_alignments(name, aln, transcript_names, weight=None, features=None,
                            platform="long", b=None) -> Library:
    """
    Collapse filtered alignments into equivalence classes.

    Reads with the same transcript set (and, if given, the same per-alignment
    features) share a class. With integer features (e.g. binned read-end
    positions) the collapse is exact for any bias model that depends on the
    features only. With real-valued weights the class weight is the mean of
    the per-read normalised weights (salmon's rich-EC approximation).

    Args:
        aln              : DataFrame from filter_alignments (columns read, tx, ...).
        transcript_names : list[str] (T,) -- target index order; alignments to
                           transcripts outside it are dropped.
        weight           : np.ndarray (n_aln,) | None -- per-alignment weight w_rt; None means 1.
        features         : np.ndarray (n_aln, F) int | None -- per-alignment features stored in
                           Library.feat and made part of the EC key.
        b                : np.ndarray (T,) | None -- sampling rates; ones by default.
    """
    pos = pd.Series(np.arange(len(transcript_names)), index=transcript_names)
    tx = pos.reindex(aln["tx"].astype(str)).to_numpy()
    keep = ~np.isnan(tx)
    tx = tx[keep].astype(np.int64)
    read = pd.factorize(aln["read"])[0][keep]
    w = np.ones(len(tx)) if weight is None else np.asarray(weight, dtype=np.float64)[keep]
    F = None if features is None else np.asarray(features, dtype=np.int64).reshape(len(aln), -1)[keep]

    order = np.lexsort((tx, read))
    read, tx, w = read[order], tx[order], w[order]
    if F is not None:
        F = F[order]
    starts = np.flatnonzero(np.r_[True, read[1:] != read[:-1]])
    sizes = np.diff(np.r_[starts, len(read)])
    w_norm = w / np.repeat(np.add.reduceat(w, starts), sizes)

    # EC key per read: bytes of its sorted transcripts (and their features)
    key_arr = tx[:, None] if F is None else np.column_stack([tx, F])
    keys = [key_arr[s:s + n].tobytes() for s, n in zip(starts, sizes)]
    ec_id, uniq = pd.factorize(pd.Series(keys), sort=False)
    E = len(uniq)
    counts = np.bincount(ec_id, minlength=E).astype(np.float64)
    first = np.full(E, -1, dtype=np.int64)
    first[ec_id[::-1]] = np.arange(len(ec_id))[::-1]
    ec_tx = [tx[starts[e]:starts[e] + sizes[e]] for e in first]
    feat = None if F is None else [F[starts[e]:starts[e] + sizes[e]] for e in first]
    c = None
    if weight is not None:
        # mean normalised weight per (EC, position within EC)
        pos_in_read = np.arange(len(tx)) - np.repeat(starts, sizes)
        ec_off = np.r_[0, np.cumsum(sizes[first])]
        flat = ec_off[np.repeat(ec_id, sizes)] + pos_in_read
        csum = np.bincount(flat, weights=w_norm, minlength=ec_off[-1])
        csum /= np.repeat(counts, sizes[first])
        c = [csum[ec_off[e]:ec_off[e + 1]] for e in range(E)]

    tcc = TCCData(ec_counts=counts, ec_transcripts=ec_tx, transcript_names=list(transcript_names),
                  total_reads=int(counts.sum()), n_targets=len(transcript_names), sample_dir=name)
    if b is None:
        b = np.ones(len(transcript_names))
    return Library(name=name, tcc=tcc, b=b, platform=platform, c=c, feat=feat)
