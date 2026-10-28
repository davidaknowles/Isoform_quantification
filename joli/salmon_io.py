"""
salmon_io.py
============
Load salmon equivalence classes (``salmon quant --dumpEq --dumpEqWeights``)
as a JointEM Library.

salmon's likelihood for fragment f is sum_t (alpha_t / l_t) w_ft, where
alpha_t is the nucleotide fraction, l_t the bias-corrected effective length
(quant.sf EffectiveLength, which absorbs --seqBias, --gcBias and --posBias)
and w_ft the "auxiliary" weight (fragment length, alignment score,
orientation). With --dumpEqWeights the rich equivalence classes store
salmon's combined weights v_et proportional to w_et / l_t, normalised within
the class (checked empirically: with c_et = v_et l_t salmon's own estimate
is a fixed point of JointEM). With molar abundance mu_t proportional to
alpha_t / l_t this is the JointEM model with b_t = l_t and c_et = v_et l_t.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .joint import Library
from .load_tcc import TCCData


def read_salmon_eq(path):
    """
    Parse eq_classes.txt[.gz].

    Returns:
        names   : list[str] (T,)          -- transcript names in salmon's order
        ec_tx   : list[np.ndarray]        -- transcript indices per EC
        ec_w    : list[np.ndarray] | None -- rich weights per EC (None if not dumped)
        counts  : np.ndarray (E,)         -- fragments per EC
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        T = int(fh.readline())
        E = int(fh.readline())
        names = [fh.readline().rstrip("\n") for _ in range(T)]
        ec_tx, ec_w = [], []
        counts = np.empty(E, dtype=np.float64)
        weighted = None
        for e in range(E):
            f = fh.readline().split()
            k = int(f[0])
            if weighted is None:
                weighted = len(f) == 2 * k + 2
            ec_tx.append(np.array(f[1:k + 1], dtype=np.int64))
            if weighted:
                ec_w.append(np.array(f[k + 1:2 * k + 1], dtype=np.float64))
            counts[e] = float(f[-1])
    return names, ec_tx, (ec_w if weighted else None), counts


def read_salmon_quant(quant_dir):
    """quant.sf as a DataFrame indexed by transcript name."""
    return pd.read_csv(Path(quant_dir) / "quant.sf", sep="\t", index_col=0)


def library_from_salmon(name, quant_dir, transcript_names=None, use_weights=True,
                        platform="short") -> Library:
    """
    Library from a salmon output directory.

    Args:
        quant_dir        : salmon -o directory (needs aux_info/eq_classes.txt[.gz] and quant.sf).
        transcript_names : list[str] | None -- target index order (e.g. a kallisto
                           transcripts.txt shared with long-read libraries); salmon's
                           order if None. Transcripts absent from salmon's index get
                           b = 1 and never appear in an EC.
        use_weights      : bool -- use rich EC weights as c (else c = 1).
    """
    quant_dir = Path(quant_dir)
    eq = quant_dir / "aux_info" / "eq_classes.txt.gz"
    if not eq.exists():
        eq = eq.with_suffix("")
    names, ec_tx, ec_w, counts = read_salmon_eq(eq)
    quant = read_salmon_quant(quant_dir)
    if transcript_names is None:
        transcript_names = names
    pos = pd.Series(np.arange(len(transcript_names)), index=transcript_names)
    remap = pos.reindex(names).to_numpy()
    if np.isnan(remap).any():
        missing = [n for n, r in zip(names, remap) if np.isnan(r)][:5]
        raise ValueError(f"salmon transcripts missing from target index, e.g. {missing}")
    remap = remap.astype(np.int64)
    b = quant["EffectiveLength"].reindex(transcript_names).fillna(1.0).to_numpy(dtype=np.float64)
    tcc = TCCData(
        ec_counts=counts,
        ec_transcripts=[remap[t] for t in ec_tx],
        transcript_names=list(transcript_names),
        total_reads=int(counts.sum()),
        n_targets=len(transcript_names),
        sample_dir=str(quant_dir),
    )
    c = None
    if use_weights and ec_w is not None:
        c = [w * b[t] for w, t in zip(ec_w, tcc.ec_transcripts)]
    return Library(name=name, tcc=tcc, b=b, platform=platform, c=c)
