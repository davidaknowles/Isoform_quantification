"""
weights.py
==========
Step 1.2 of JOLI-Kallisto Phase 1: compute effective lengths and per-EC
transcript weights for the EM algorithm.

Mirrors kallisto's weights.cpp logic. Three modes:

  "uniform"  (long reads, Phase 1 default):
      eff_lens[t] = 1.0 for all transcripts.
      Equivalent to kallisto's behaviour when no fragment length distribution
      is available (--long mode on TCC-only input).

  "kallisto" (short reads, with pre-computed joli_efflen.txt):
      eff_lens loaded from joli_efflen.txt produced by Step 3.5 (backward compat).

  "fld" (short reads, preferred — no kallisto quant call needed):
      eff_lens computed in Python from:
        - flens.txt  : FLD histogram (1000 values) written by kallisto bus --paired
        - FASTA file : transcript lengths parsed once from transcriptome.fasta
      Matches kallisto's MinCollector::compute_mean_frag_lens_trunc() +
      weights::calc_eff_lens() exactly.

EC weights:
  For each EC, for each transcript t in that EC:
      ec_weights[ec][i] = 1.0 / eff_lens[t]
  where i is the position of t within the EC's transcript list.

  This matches kallisto's weight_map_ construction in weights.cpp.

Inputs:
  - TCCData   : from load_tcc.py
  - transcript_lengths : optional np.ndarray, shape (n_transcripts,) [bp]
  - mean_frag_len      : optional float, mean fragment length [bp]
  - flens              : optional np.ndarray, shape (n_transcripts,) -- pre-computed
                         eff_lens from joli_efflen.txt (backward compat)
  - fld                : optional np.ndarray, shape (1000,) -- FLD histogram from
                         flens.txt (kallisto bus --paired output)
  - mode               : "uniform" | "kallisto" | "fld"

Outputs:
  - WeightData dataclass:
      eff_lens    : np.ndarray, shape (n_transcripts,) -- effective length per tx
      ec_weights  : list[np.ndarray]                  -- ec_weights[ec][i] = weight
                                                          for i-th tx in that EC
"""

import os
from dataclasses import dataclass

import numpy as np

from .load_tcc import TCCData


# ============================================================
# FLD constants — match kallisto exactly
# ============================================================

MAX_FRAG_LEN = 1000  # size of FLD histogram (kallisto constant)


# ============================================================
# Data container
# ============================================================

@dataclass
class WeightData:
    """
    Container for computed effective lengths and EC-level transcript weights.

    Attributes:
        eff_lens   (np.ndarray, float64, shape [n_transcripts]):
                    Effective length for each transcript. In "uniform" mode
                    all values are 1.0. In "kallisto" mode: max(1, len - mfl + 1).

        ec_weights (list[np.ndarray], length n_ecs):
                    ec_weights[ec_id] is a float64 array of length
                    len(ec_transcripts[ec_id]). Element i is the weight
                    (1/eff_len) for the i-th transcript in that EC.
                    This is the direct input to the EM E-step denominator.
    """
    eff_lens: np.ndarray
    ec_weights: list

    def __repr__(self) -> str:
        n_tx = len(self.eff_lens) if self.eff_lens is not None else 0
        n_ec = len(self.ec_weights) if self.ec_weights is not None else 0
        return f"WeightData(n_transcripts={n_tx}, n_ecs={n_ec})"


# ============================================================
# FLD-based effective length computation (short reads, "fld" mode)
# Mirrors kallisto MinCollector::compute_mean_frag_lens_trunc() +
#         weights::get_frag_len_means() + weights::calc_eff_lens()
# ============================================================

def load_fld_histogram(flens_path: str) -> np.ndarray:
    """
    Load the FLD histogram written by kallisto bus --paired for short reads.

    Format: one line, space-separated unsigned integers, MAX_FRAG_LEN (1000) values.
    flens[i] = number of read pairs whose insert size equals i.

    Args:
        flens_path : str -- Path to flens.txt (from kallisto bus --paired output dir).

    Returns:
        np.ndarray (uint32, shape [MAX_FRAG_LEN=1000]) -- FLD histogram.

    Raises:
        FileNotFoundError : If flens.txt does not exist.
        ValueError        : If the file does not contain exactly MAX_FRAG_LEN values.
    """
    if not os.path.exists(flens_path):
        raise FileNotFoundError(
            f"flens.txt not found: {flens_path}. "
            "It is written by 'kallisto bus --paired' for short paired-end reads."
        )
    with open(flens_path) as fh:
        values = np.array(fh.read().split(), dtype=np.uint32)
    if len(values) != MAX_FRAG_LEN:
        raise ValueError(
            f"Expected {MAX_FRAG_LEN} values in flens.txt, got {len(values)}. "
            "Ensure this is the FLD histogram from kallisto bus --paired, "
            "not joli_efflen.txt (per-transcript eff_lens)."
        )
    total = int(values.sum())
    print(f"  Loaded FLD histogram: {MAX_FRAG_LEN} bins, {total} total reads in FLD")
    return values


def compute_mean_fl_trunc(fld: np.ndarray) -> np.ndarray:
    """
    Compute the length-conditional cumulative mean of the FLD.

    Mirrors MinCollector::compute_mean_frag_lens_trunc() from kallisto.

    mean_fl_trunc[i] = E[fragment_length | fragment_length <= i]
                     = sum_{j=1}^{i}(fld[j]*j) / sum_{j=1}^{i} fld[j]

    For long transcripts (i >> global_mean), mean_fl_trunc[i] ≈ global mean.
    For short transcripts (i < global_mean), mean_fl_trunc[i] < global mean,
    so their effective length is correctly shrunk.

    Args:
        fld : np.ndarray (uint32 or float64, shape [MAX_FRAG_LEN]) -- FLD histogram.

    Returns:
        np.ndarray (float64, shape [MAX_FRAG_LEN]) -- Length-conditional cumulative means.
        mean_fl_trunc[0] = 0.0 (undefined for length 0).
    """
    # Vectorized cumulative sum — equivalent to the C++ loop:
    #   mass[i]   = fld[i]*i + mass[i-1]   → cumsum(fld * indices)
    #   counts[i] = fld[i]  + counts[i-1]  → cumsum(fld)
    indices = np.arange(MAX_FRAG_LEN, dtype=np.float64)
    mass    = np.cumsum(fld.astype(np.float64) * indices)   # cumulative weighted sum
    counts  = np.cumsum(fld.astype(np.float64))             # cumulative count

    safe_counts   = np.where(counts > 0, counts, 1.0)   # avoid 0/0
    mean_fl_trunc = np.where(counts > 0, mass / safe_counts, 0.0)
    mean_fl_trunc[0] = 0.0  # length 0 is undefined

    global_mean = float(mean_fl_trunc[MAX_FRAG_LEN - 1])
    print(f"  FLD global mean (mean_fl_trunc[999]): {global_mean:.2f} bp")
    return mean_fl_trunc


def calc_eff_lens_from_fld(
    transcript_lengths: np.ndarray,
    mean_fl_trunc: np.ndarray,
) -> np.ndarray:
    """
    Compute per-transcript effective lengths from the truncated FLD means.

    Mirrors get_frag_len_means() + calc_eff_lens() from kallisto weights.cpp.

    For each transcript of length L:
        fl_mean = mean_fl_trunc[min(L, MAX_FRAG_LEN - 1)]
        eff_len = L - fl_mean + 1.0
        if eff_len < 1.0: eff_len = L  (fallback: very short transcripts)

    Args:
        transcript_lengths : np.ndarray (int or float64, shape [n_transcripts]) -- bp.
        mean_fl_trunc      : np.ndarray (float64, shape [MAX_FRAG_LEN]) -- from
                             compute_mean_fl_trunc().

    Returns:
        np.ndarray (float64, shape [n_transcripts]) -- Effective lengths.
    """
    lengths = transcript_lengths.astype(np.float64)
    # Cap index at MAX_FRAG_LEN - 1 for transcripts longer than the FLD window
    idx     = np.minimum(lengths.astype(np.int64), MAX_FRAG_LEN - 1)
    fl_mean = mean_fl_trunc[idx]

    raw_eff = lengths - fl_mean + 1.0
    # Fallback: if eff_len < 1 (transcript shorter than mean fragment), use raw length
    eff_lens = np.where(raw_eff >= 1.0, raw_eff, lengths)

    n_fallback = int((raw_eff < 1.0).sum())
    print(f"  eff_lens from FLD: min={eff_lens.min():.1f}, max={eff_lens.max():.1f}, "
          f"mean={eff_lens.mean():.1f}, fallback_count={n_fallback}")
    return eff_lens.astype(np.float64)


def load_fasta_transcript_lengths(
    fasta_path: str,
    transcript_names: list,
) -> np.ndarray:
    """
    Parse transcript lengths from a FASTA file, returning them in transcript_names order.

    The transcript ID is the first whitespace-delimited token after '>':
      ">NR_046018.2 gene_id:DDX11L1 ..."  →  ID = "NR_046018.2"

    This matches the format produced by the same index used for kallisto bus.

    Args:
        fasta_path       : str       -- Path to transcriptome.fasta.
        transcript_names : list[str] -- Ordered transcript names (from transcripts.txt).

    Returns:
        np.ndarray (int64, shape [n_transcripts]) -- Length in bp per transcript,
        in the same order as transcript_names.

    Raises:
        FileNotFoundError : If fasta_path does not exist.
        ValueError        : If any transcript in transcript_names is absent from the FASTA.
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Transcriptome FASTA not found: {fasta_path}")

    print(f"  Parsing transcript lengths from FASTA: {fasta_path}")

    # One-pass scan: accumulate sequence length per transcript ID
    lengths_dict: dict = {}
    current_id: str | None = None
    current_len: int = 0

    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith('>'):
                if current_id is not None:
                    lengths_dict[current_id] = current_len
                current_id  = line[1:].split()[0]   # first token after '>'
                current_len = 0
            else:
                current_len += len(line.rstrip('\n'))
    if current_id is not None:
        lengths_dict[current_id] = current_len

    print(f"  Parsed {len(lengths_dict)} transcripts from FASTA")

    # Build ordered array matching transcript_names
    lengths = np.zeros(len(transcript_names), dtype=np.int64)
    missing = []
    for i, name in enumerate(transcript_names):
        if name in lengths_dict:
            lengths[i] = lengths_dict[name]
        else:
            missing.append(name)

    if missing:
        raise ValueError(
            f"{len(missing)} transcripts in transcripts.txt not found in FASTA "
            f"(first 5: {missing[:5]}). "
            "Ensure transcriptome.fasta matches the kallisto index."
        )

    print(f"  Transcript lengths: min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}")
    return lengths


# ============================================================
# Effective length computation
# ============================================================

def _compute_eff_lens_uniform(n_transcripts: int) -> np.ndarray:
    """
    Uniform mode: every transcript gets effective length 1.0.

    Equivalent to treating all transcripts equally, which matches kallisto's
    behaviour when no fragment length distribution (FLD) is provided for
    long-read TCC input.

    Args:
        n_transcripts : int -- Total number of transcripts.

    Returns:
        np.ndarray (float64, shape [n_transcripts]) -- all ones.
    """
    return np.ones(n_transcripts, dtype=np.float64)


def _compute_eff_lens_kallisto(
    transcript_lengths: np.ndarray,
    mean_frag_len: float,
) -> np.ndarray:
    """
    Kallisto mode: eff_len[t] = max(1.0, length[t] - mean_frag_len + 1).

    Guard (matches kallisto weights.cpp line ~70):
      if length[t] - mean_frag_len + 1 < 1.0:
          eff_lens[t] = length[t]   (use raw length as fallback)

    Args:
        transcript_lengths : np.ndarray (float64, shape [n_transcripts]) -- lengths in bp.
        mean_frag_len      : float -- Mean fragment length in bp.

    Returns:
        np.ndarray (float64, shape [n_transcripts]) -- Effective lengths.
    """
    raw = transcript_lengths - mean_frag_len + 1.0
    eff = np.where(raw >= 1.0, raw, transcript_lengths)
    return eff.astype(np.float64)


# ============================================================
# EC weight construction
# ============================================================

def _build_ec_weights(
    ec_transcripts: list,
    eff_lens: np.ndarray,
) -> list:
    """
    Build per-EC weight arrays from effective lengths.

    For each EC, creates a float64 array where element i =
    1.0 / eff_lens[ec_transcripts[ec][i]].

    This is the weight used in the EM E-step:
      denom = sum( theta[t] * ec_weights[ec][i]  for i, t in enumerate(EC) )

    Args:
        ec_transcripts : list[list[int]] -- transcript indices per EC.
        eff_lens       : np.ndarray      -- effective length per transcript.

    Returns:
        list[np.ndarray] -- ec_weights[ec_id][i] = 1 / eff_lens[tx_i].
    """
    ec_weights = []
    for txs in ec_transcripts:
        # Gather effective lengths for transcripts in this EC, then invert
        lens = eff_lens[txs]          # shape (len(txs),)
        weights = 1.0 / lens          # element-wise reciprocal
        ec_weights.append(weights)
    return ec_weights


# ============================================================
# Main entry point
# ============================================================

def compute_weights(
    tcc_data: TCCData,
    transcript_lengths: np.ndarray = None,
    mean_frag_len: float = None,
    flens: np.ndarray = None,
    fld: np.ndarray = None,
    mode: str = "uniform",
) -> WeightData:
    """
    Compute effective lengths and EC transcript weights.

    Args:
        tcc_data           : TCCData    -- Loaded TCC data (from load_tcc.py).
        transcript_lengths : np.ndarray -- Shape (n_transcripts,), lengths in bp.
                                          Required for mode="fld" and legacy "kallisto".
        mean_frag_len      : float      -- Mean fragment length in bp.
                                          Used for legacy mode="kallisto" path.
        flens              : np.ndarray -- Shape (n_transcripts,), pre-computed eff_lens
                                          from joli_efflen.txt (backward compat).
                                          When provided for mode="kallisto", takes
                                          priority over transcript_lengths/mean_frag_len.
        fld                : np.ndarray -- Shape (MAX_FRAG_LEN=1000,), FLD histogram
                                          from kallisto bus --paired flens.txt.
                                          Required for mode="fld".
        mode               : str        -- "uniform"  : eff_len=1.0 (long reads, Phase 1)
                                           "kallisto" : from joli_efflen.txt (backward compat)
                                           "fld"      : Python FLD computation (preferred for
                                                        short reads — no kallisto quant needed)

    Returns:
        WeightData -- Contains eff_lens and ec_weights arrays.

    Raises:
        ValueError : If required inputs are missing for the chosen mode.
        ValueError : If provided arrays length doesn't match n_transcripts.
    """
    n_transcripts = len(tcc_data.transcript_names)
    print(f"\n[weights] Computing weights (mode='{mode}', "
          f"n_transcripts={n_transcripts}, n_ecs={len(tcc_data.ec_transcripts)})")

    # --- Compute effective lengths ---
    if mode == "uniform":
        eff_lens = _compute_eff_lens_uniform(n_transcripts)
        print(f"  Uniform mode: eff_lens = 1.0 for all {n_transcripts} transcripts")

    elif mode == "fld":
        # Python FLD path: compute eff_lens from the FLD histogram + transcript lengths.
        # Mirrors kallisto compute_mean_frag_lens_trunc() + calc_eff_lens() exactly.
        # No kallisto quant call needed — flens.txt is written by kallisto bus --paired.
        if fld is None:
            raise ValueError("mode='fld' requires fld=<np.ndarray from flens.txt (1000 values)>.")
        if transcript_lengths is None:
            raise ValueError(
                "mode='fld' requires transcript_lengths=<np.ndarray from FASTA>."
            )
        if len(fld) != MAX_FRAG_LEN:
            raise ValueError(
                f"fld has {len(fld)} values; expected {MAX_FRAG_LEN} (MAX_FRAG_LEN)."
            )
        if len(transcript_lengths) != n_transcripts:
            raise ValueError(
                f"transcript_lengths has {len(transcript_lengths)} entries "
                f"but n_transcripts={n_transcripts}."
            )
        mean_fl_trunc = compute_mean_fl_trunc(fld)
        eff_lens      = calc_eff_lens_from_fld(transcript_lengths, mean_fl_trunc)

    elif mode == "kallisto":
        if flens is not None:
            # Backward-compat path: eff_lens loaded from joli_efflen.txt.
            if len(flens) != n_transcripts:
                raise ValueError(
                    f"flens has {len(flens)} entries but n_transcripts={n_transcripts}."
                )
            eff_lens = flens.astype(np.float64)
            sentinel = 4294967295.0  # UINT32_MAX — unobserved transcripts
            n_sentinel = int((eff_lens >= sentinel).sum())
            print(f"  Kallisto mode (joli_efflen.txt): pre-computed eff_lens loaded")
            print(f"  eff_lens: min={eff_lens[eff_lens < sentinel].min():.1f}, "
                  f"max={eff_lens[eff_lens < sentinel].max():.1f}, "
                  f"sentinel (unobserved): {n_sentinel}")
        elif transcript_lengths is not None and mean_frag_len is not None:
            # Legacy path: single global mean_frag_len (less accurate than "fld" mode).
            if len(transcript_lengths) != n_transcripts:
                raise ValueError(
                    f"transcript_lengths has {len(transcript_lengths)} entries "
                    f"but n_transcripts={n_transcripts}."
                )
            eff_lens = _compute_eff_lens_kallisto(
                transcript_lengths.astype(np.float64), float(mean_frag_len)
            )
            print(f"  Kallisto mode (legacy): mean_frag_len={mean_frag_len:.1f}")
            print(f"  eff_lens: min={eff_lens.min():.1f}, "
                  f"max={eff_lens.max():.1f}, mean={eff_lens.mean():.1f}")
        else:
            raise ValueError(
                "mode='kallisto' requires either:\n"
                "  flens=<np.ndarray from joli_efflen.txt>  (backward compat), or\n"
                "  transcript_lengths + mean_frag_len (legacy formula).\n"
                "For new short-read pipelines, prefer mode='fld'."
            )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'uniform', 'fld', or 'kallisto'.")

    # --- Build EC weight arrays ---
    ec_weights = _build_ec_weights(tcc_data.ec_transcripts, eff_lens)

    # Checkpoint: verify shapes
    assert len(ec_weights) == len(tcc_data.ec_transcripts), "EC weight count mismatch"
    for ec_id, (txs, w) in enumerate(zip(tcc_data.ec_transcripts, ec_weights)):
        assert len(w) == len(txs), \
            f"EC {ec_id}: weight array length {len(w)} != transcript count {len(txs)}"

    print(f"  EC weights computed for {len(ec_weights)} ECs")
    print(f"[weights] Done.")

    return WeightData(eff_lens=eff_lens, ec_weights=ec_weights)
