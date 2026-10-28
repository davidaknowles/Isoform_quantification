"""
em_algorithm.py
===============
Step 1.3 of JOLI-Kallisto Phase 1: EC-based EM algorithm.

Implements JoliEM — a vectorized EM that operates on equivalence classes (ECs)
rather than individual reads. Mathematically equivalent to kallisto's C++ EM
(EMAlgorithm.h) but written in Python/NumPy, and designed to extend to
Dirichlet MAP (Phase 2), multi-sample GD (Phase 3), and VI (Phase 4).

Single-transcript EC handling:
  Single-tx ECs have deterministic assignment (probability = 1.0) so the E-step
  does not process them.  Their per-transcript read counts are pre-computed in
  _preprocess() as _single_tx_counts_per_tx (a constant vector) and added to the
  M-step numerator every round alongside the multi-tx expected counts.  This means
  the E-step probability P(read → t | multi-tx EC) correctly reflects each
  transcript's total abundance (single-tx + multi-tx), not just its multi-tx share.
  The final alpha is theta * total_all_reads — no separate post-convergence addition.

Variable mapping to kallisto EMAlgorithm.h:
  alpha       <->  alpha_          (raw expected counts; NOT normalized)
  theta       <->  (internal only) normalized abundances used for convergence
  n           <->  next_alpha      (accumulated expected counts per M-step)
  ec_counts   <->  counts_         (read count per EC)
  ec_weights  <->  weight_map_     (1/eff_len per transcript per EC)
  TOLERANCE   <->  TOLERANCE       (numeric floor for denominator)

Variable mapping to AT_code EM_VIorMAP_GD_vector.py:
  alpha       <->  all_theta[sample] (after scaling back to raw counts)
  n           <->  all_n[sample]
  phi         <->  all_Phi_ri      (fractional assignment; implicit here via n)

Inputs:
  - TCCData   (from load_tcc.py)
  - WeightData (from weights.py)
  - EM hyperparameters (max_em_rounds, min_rounds, convergence thresholds)

Outputs:
  - EMResult dataclass:
      alpha      : np.ndarray (float64, shape [n_transcripts]) -- raw expected counts
                   (multi-tx EM counts + single-tx raw counts; does NOT sum to 1)
      n_rounds   : int  -- number of EM iterations completed
      converged  : bool -- whether the convergence criterion was met
"""

from dataclasses import dataclass

import numpy as np

from .load_tcc import TCCData
from .weights import WeightData


# ============================================================
# Constants — match kallisto EMAlgorithm.h exactly
# ============================================================

TOLERANCE          = np.finfo(np.float64).tiny   # ~2.2e-308; floor for denominators
ALPHA_LIMIT        = 1e-7    # transcripts below this × 0.1 are zeroed out
ALPHA_CHANGE_LIMIT = 1e-2    # convergence threshold — interpreted differently per mode:
                             #   "kallisto" mode: raw expected counts > 0.01  (matches C++ EM)
                             #   "joli"    mode: normalized theta   > 0.01  (fast, for MAP/VI)
ALPHA_CHANGE       = 1e-2    # 1% relative change threshold for convergence
DEFAULT_MIN_ROUNDS = 50      # minimum EM iterations before convergence is checked


# ============================================================
# Result container
# ============================================================

@dataclass
class EMResult:
    """
    Output from one run of JoliEM.

    Attributes:
        theta_unnorm (np.ndarray, float64, shape [n_transcripts]):
                   Raw expected read counts per transcript. Multi-tx EM expected
                   counts + single-tx raw counts added post-convergence.
                   Does NOT sum to 1. Matches kallisto's alpha_ after run().
                   Use theta_unnorm / eff_lens → rho → TPM for output (see output_writer.py).
                   Named theta_unnorm to distinguish from the Dirichlet alpha_prior.

        n_rounds  (int): Number of EM iterations completed.

        converged (bool): True if the convergence criterion was satisfied before
                          reaching max_em_rounds.

        snapshots (list | None): When snapshot_interval > 0 in run(), a list of
                   (round_num, theta) tuples saved every snapshot_interval rounds
                   plus the final theta. theta is normalized (sums to ~1).
                   None when snapshots are disabled (snapshot_interval=0).
    """
    theta_unnorm: np.ndarray
    n_rounds:     int
    converged:    bool
    snapshots:    list = None


# ============================================================
# JoliEM class
# ============================================================

class JoliEM:
    """
    EC-based EM algorithm for isoform quantification.

    The E-step and M-step are combined in each iteration (same structure as
    kallisto's EMAlgorithm.h::run()):

      For each EC with count > 0:
        denom = sum_t ( theta[t] * ec_weight[t] )
        For each t in EC:
          n[t] += ec_count * theta[t] * ec_weight[t] / denom

    Single-transcript ECs are handled deterministically (no division needed),
    matching kallisto's optimization for singleton ECs.

    Multi-transcript ECs are vectorized using pre-built flat index arrays to
    avoid a Python loop over EC positions. See _preprocess() for details.
    """

    def __init__(self, tcc_data: TCCData, weight_data: WeightData,
                 em_include_single_tx: bool = True):
        """
        Pre-process TCC and weight data into vectorization-friendly flat arrays.

        Args:
            tcc_data             : TCCData    -- Parsed bustools output.
            weight_data          : WeightData -- Effective lengths and EC weights.
            em_include_single_tx : bool       -- When True, single-tx read counts are added
                                                to the M-step numerator every round so the
                                                E-step for multi-tx ECs reflects total
                                                abundance (single-tx + multi-tx).
                                                Matches kallisto's SHORT-READ EM path
                                                (EMAlgorithm.h: single-tx set in next_alpha
                                                at the start of every round).
                                                When False, single-tx excluded from EM loop,
                                                added post-convergence only — matches
                                                kallisto's LONG-READ EM path.
        """
        self.n_transcripts         = len(tcc_data.transcript_names)
        self._em_include_single_tx = em_include_single_tx
        self.ec_counts      = tcc_data.ec_counts          # shape (n_ecs,)
        self.ec_transcripts = tcc_data.ec_transcripts     # list[list[int]]
        self.eff_lens       = weight_data.eff_lens        # shape (n_transcripts,)
        self.ec_weights     = weight_data.ec_weights      # list[np.ndarray]

        self._preprocess()

    def _preprocess(self):
        """
        Split ECs into single-transcript and multi-transcript groups.
        Build flat index arrays for vectorized multi-tx EM computation.

        Single-tx ECs: deterministic assignment, handled with np.add.at once.
        Multi-tx ECs:  fractional assignment, vectorized via flat arrays.

        Flat arrays built here (all parallel, indexed by EC position):
          _multi_flat_tx      : transcript index for each (ec, position) pair
          _multi_flat_weights : weight for each (ec, position) pair
          _multi_flat_ec_idx  : which multi-EC each position belongs to
          _multi_ec_counts    : count for each multi-EC (one per EC, not per position)
        """
        # --- Separate single-tx and multi-tx ECs ---
        single_ec_ids = []
        multi_ec_ids  = []
        for ec_id, txs in enumerate(self.ec_transcripts):
            if len(txs) == 1:
                single_ec_ids.append(ec_id)
            else:
                multi_ec_ids.append(ec_id)

        # Single-tx: static arrays (independent of theta, computed once)
        if single_ec_ids:
            single_ec_ids = np.array(single_ec_ids, dtype=np.int64)
            self._single_tx_ids    = np.array(
                [self.ec_transcripts[i][0] for i in single_ec_ids], dtype=np.int64)
            self._single_ec_counts = self.ec_counts[single_ec_ids]  # shape (n_single,)
        else:
            self._single_tx_ids    = np.array([], dtype=np.int64)
            self._single_ec_counts = np.array([], dtype=np.float64)

        # Multi-tx: build flat arrays
        flat_tx      = []
        flat_weights = []
        flat_ec_idx  = []
        multi_counts = []

        for local_idx, ec_id in enumerate(multi_ec_ids):
            txs     = self.ec_transcripts[ec_id]
            weights = self.ec_weights[ec_id]           # np.ndarray, shape (len(txs),)
            for pos, (tx, w) in enumerate(zip(txs, weights)):
                flat_tx.append(tx)
                flat_weights.append(w)
                flat_ec_idx.append(local_idx)
            multi_counts.append(self.ec_counts[ec_id])

        if flat_tx:
            self._multi_flat_tx      = np.array(flat_tx,      dtype=np.int64)
            self._multi_flat_weights = np.array(flat_weights,  dtype=np.float64)
            self._multi_flat_ec_idx  = np.array(flat_ec_idx,   dtype=np.int64)
            self._multi_ec_counts    = np.array(multi_counts,   dtype=np.float64)
            self._n_multi_ecs        = len(multi_ec_ids)
        else:
            self._multi_flat_tx      = np.array([], dtype=np.int64)
            self._multi_flat_weights = np.array([], dtype=np.float64)
            self._multi_flat_ec_idx  = np.array([], dtype=np.int64)
            self._multi_ec_counts    = np.array([], dtype=np.float64)
            self._n_multi_ecs        = 0

        # Active-transcript mask: True only for transcripts that appear in at
        # least one EC for this sample. Transcripts with no reads must stay
        # zero even when alpha_prior > 0 (prior must not invent abundance).
        self._active_tx_mask = np.zeros(self.n_transcripts, dtype=bool)
        if len(self._single_tx_ids) > 0:
            self._active_tx_mask[self._single_tx_ids] = True
        if len(self._multi_flat_tx) > 0:
            self._active_tx_mask[self._multi_flat_tx] = True
        n_active = int(self._active_tx_mask.sum())

        # Total multi-tx read count — fixed (depends only on data, not theta).
        # Cached here so both run() and em_step() can use it without recomputing.
        self._total_multi_reads = (
            float(self._multi_ec_counts.sum()) if len(self._multi_ec_counts) > 0 else 0.0
        )

        # Per-transcript sum of single-tx read counts (constant across all EM rounds).
        # Used by run() to include single-tx evidence in the M-step numerator so that
        # transcripts with strong single-tx support correctly influence the E-step
        # probability when they also appear in multi-tx ECs.
        self._single_tx_counts_per_tx = np.zeros(self.n_transcripts, dtype=np.float64)
        if len(self._single_tx_ids) > 0:
            np.add.at(self._single_tx_counts_per_tx, self._single_tx_ids,
                      self._single_ec_counts.astype(np.float64))
        self._total_single_reads = float(self._single_tx_counts_per_tx.sum())
        self._total_all_reads    = self._total_multi_reads + self._total_single_reads

        print(f"[JoliEM] Pre-processed: {len(single_ec_ids)} single-tx ECs, "
              f"{self._n_multi_ecs} multi-tx ECs, "
              f"{len(self._multi_flat_tx)} total EC-transcript positions, "
              f"{n_active} active transcripts (have reads in this sample), "
              f"total_reads={self._total_all_reads:.0f} "
              f"({self._total_single_reads:.0f} single-tx + {self._total_multi_reads:.0f} multi-tx)")

    def _em_step(self, theta: np.ndarray) -> np.ndarray:
        """
        Run one combined E+M step and return updated n (expected counts per tx).

        Only multi-transcript ECs are processed here.  Single-tx ECs have
        deterministic assignment; their counts are held in _single_tx_counts_per_tx
        and added by the caller (run() M-step) every round.

        The E-step computes fractional assignment weights; the M-step accumulates
        them into n[t]. Both happen in one pass, matching kallisto's structure.

        Args:
            theta : np.ndarray (float64, shape [n_transcripts]) -- current normalized
                    abundances (derived from multi-tx EM only; excludes single-tx).

        Returns:
            np.ndarray (float64, shape [n_transcripts]) -- n[t] = expected counts
            from multi-tx ECs only.
        """
        n = np.zeros(self.n_transcripts, dtype=np.float64)

        # --- Multi-transcript ECs: vectorized fractional assignment ---
        # Single-tx ECs are intentionally excluded here; added post-convergence.
        if self._n_multi_ecs > 0:
            # theta values and weights for every (ec, position) in flat arrays
            theta_per_pos   = theta[self._multi_flat_tx]              # shape (n_positions,)
            weighted_theta  = theta_per_pos * self._multi_flat_weights # theta[t] * w[t,ec]

            # Denominator per multi-EC: sum of weighted_theta over positions in each EC
            denominators = np.zeros(self._n_multi_ecs, dtype=np.float64)
            np.add.at(denominators, self._multi_flat_ec_idx, weighted_theta)

            # Broadcast denominator and EC count back to each position
            denom_per_pos = denominators[self._multi_flat_ec_idx]    # shape (n_positions,)
            count_per_pos = self._multi_ec_counts[self._multi_flat_ec_idx]

            # Mask positions where denominator is too small (avoids div-by-zero)
            valid = denom_per_pos >= TOLERANCE
            contributions = np.where(
                valid,
                count_per_pos * weighted_theta / np.where(valid, denom_per_pos, 1.0),
                0.0
            )

            # Accumulate contributions into n
            np.add.at(n, self._multi_flat_tx, contributions)

        return n

    def run(
        self,
        max_em_rounds: int      = 10000,
        min_rounds: int         = DEFAULT_MIN_ROUNDS,
        convergence_mode: str   = "kallisto",
        alpha_prior: np.ndarray = None,
        init_theta: np.ndarray  = None,
        min_read_support: float = 0.0,
        snapshot_interval: int  = 0,
    ) -> EMResult:
        """
        Run EM until convergence or max_em_rounds, then zero small abundances.

        Two convergence modes (set via convergence_mode):

          "kallisto" (default for plain EM):
            Matches kallisto EMAlgorithm.h exactly. The convergence threshold
            ALPHA_CHANGE_LIMIT is applied to RAW expected counts (alpha = theta *
            total_multi_reads), so even transcripts with tiny fractional counts
            (> 0.01 reads) are monitored. This forces JK to run until ALL small
            transcripts have truly converged, matching kallisto's isoform set.
            Also runs one extra EM round after zeroing (finalRound mechanism from
            kallisto), allowing reads to redistribute away from zeroed transcripts.

          "joli" (recommended for MAP / VI):
            Applies ALPHA_CHANGE_LIMIT to NORMALIZED theta (sums to 1). Only
            transcripts with theta > 0.01 (> 1% of total reads) are monitored.
            Converges faster. Small transcripts are left to the Dirichlet prior
            in MAP/VI mode rather than being driven to zero by more EM rounds.

        Fix A — gating the Dirichlet prior on read support (min_read_support):
            When min_read_support > 0.0, the prior is only applied to transcripts
            where the E-step produced n[t] >= min_read_support expected reads.
            Transcripts below the threshold get plain EM (no prior) — plain EM
            naturally drives multi-mapping leakage transcripts toward zero.
            When min_read_support = 0.0 (default), the prior is always applied
            (original behaviour, Fix A disabled).

        Args:
            max_em_rounds     : int         -- Maximum EM iterations (default: 10000).
            min_rounds        : int         -- Minimum iterations before convergence check
                                              (default: 50).
            convergence_mode  : str         -- "kallisto" (raw count threshold, matches LK)
                                              or "joli" (normalized theta threshold, faster).
            alpha_prior       : np.ndarray  -- Optional Dirichlet concentration vector,
                                              shape [n_transcripts]. When provided, the
                                              M-step uses the posterior mean:
                                                theta_new = (n + alpha) / sum(n + alpha)
                                              instead of plain EM (n / sum(n)).
                                              Pass None for plain EM (default).
            init_theta        : np.ndarray  -- Optional starting theta, shape [n_transcripts].
                                              When provided (e.g. carried over from a previous
                                              GD round), EM starts from this theta instead of
                                              uniform. Pass None to start from uniform (default).
            min_read_support  : float       -- Fix A flag. Minimum expected read count n[t]
                                              required to apply alpha_prior in the M-step.
                                              0.0 = disabled (prior always applied).
                                              Typical value to try: 0.1 reads.
            snapshot_interval : int         -- Save a (round_num, theta) snapshot every
                                              this many rounds. 0 = disabled (default).
                                              Used by plot_convergence_animation.py to
                                              animate convergence across training rounds.

        Returns:
            EMResult -- theta_unnorm (raw expected counts), n_rounds, converged, snapshots.
        """
        if convergence_mode not in ("kallisto", "joli"):
            raise ValueError(
                f"convergence_mode must be 'kallisto' or 'joli', got '{convergence_mode}'"
            )

        # Validate alpha_prior shape if provided
        if alpha_prior is not None:
            if alpha_prior.shape != (self.n_transcripts,):
                raise ValueError(
                    f"alpha_prior shape {alpha_prior.shape} does not match "
                    f"n_transcripts={self.n_transcripts}"
                )

        # total_all_reads: denominator for M-step normalisation and convergence scaling.
        # When em_include_single_tx=True, this covers all reads (single-tx + multi-tx).
        # When em_include_single_tx=False, single-tx is excluded from the EM loop
        # (original kallisto behaviour); the denominator is multi-tx reads only.
        total_all_reads   = self._total_all_reads if self._em_include_single_tx else self._total_multi_reads
        total_multi_reads = self._total_multi_reads  # kept for finalRound (kallisto mode)

        print(f"\n[JoliEM] Starting EM: max_rounds={max_em_rounds}, "
              f"min_rounds={min_rounds}, convergence_mode={convergence_mode}, "
              f"mode={'MAP(posterior_mean)' if alpha_prior is not None else 'plain'}, "
              f"n_transcripts={self.n_transcripts}, "
              f"total_all_reads={total_all_reads:.0f} "
              f"({self._total_single_reads:.0f} single-tx + {total_multi_reads:.0f} multi-tx), "
              f"warm_start={'yes' if init_theta is not None else 'no'}")

        # --- Initialization ---
        # Use provided init_theta (warm-start from previous GD round) or uniform.
        if init_theta is not None:
            theta = init_theta.copy().astype(np.float64)
            total_init = theta.sum()
            if total_init > 0:
                theta /= total_init     # re-normalize for safety
            else:
                theta = np.full(self.n_transcripts, 1.0 / self.n_transcripts,
                                dtype=np.float64)
        else:
            # Uniform initialization — matches kallisto's alpha_[i] = 1/num_targets
            theta = np.full(self.n_transcripts, 1.0 / self.n_transcripts,
                            dtype=np.float64)
        converged = False

        # Snapshot list: populated when snapshot_interval > 0
        snapshots = [] if snapshot_interval > 0 else None

        # Record initialization snapshot (round=-1) before any EM step.
        # theta at this point is uniform 1/T.
        if snapshots is not None:
            snapshots.append((-1, theta.copy()))

        for round_num in range(max_em_rounds):
            # One E+M step — returns expected counts from multi-tx ECs only
            n = self._em_step(theta)

            # Add fixed single-tx counts when the switch is on so the M-step
            # normalisation uses all reads, correctly informing the E-step probability.
            # When the switch is off, n_total = n (original kallisto behaviour).
            n_total = (n + self._single_tx_counts_per_tx) if self._em_include_single_tx else n

            # M-step: posterior-mean EM (with Dirichlet prior) or plain EM
            if alpha_prior is not None:
                if min_read_support > 0.0:
                    # Fix A: gate the prior on read support.
                    # Use n_total so single-tx evidence counts toward the support threshold.
                    has_support = n_total >= min_read_support
                    numerator = np.where(has_support, n_total + alpha_prior, n_total)
                else:
                    numerator = n_total + alpha_prior
            else:
                numerator = n_total

            # Zero out transcripts with no reads in this sample before normalizing.
            # Without this, alpha_prior > 0 would assign positive theta to transcripts
            # that have zero reads here, inflating nonzero-transcript counts.
            numerator[~self._active_tx_mask] = 0.0

            total = numerator.sum()
            if total > 0:
                theta_new = numerator / total
            else:
                theta_new = theta.copy()

            # --- Convergence check ---
            if convergence_mode == "kallisto":
                # Compare raw expected counts against 0.01 reads.
                # Use total_all_reads (not total_multi_reads) because theta now
                # normalises over all reads (single-tx + multi-tx).
                counts_new = theta_new * total_all_reads
                counts_old = theta     * total_all_reads
                # Guard: numpy & evaluates both sides before combining, so
                # the relative-change division runs for ALL elements including
                # those where counts_new ~ TOLERANCE (~2e-308), causing
                # overflow to inf.  Use np.where to skip the division where
                # counts_new <= ALPHA_CHANGE_LIMIT (0.01) — those elements
                # are never counted as "changed" anyway.
                monitored = counts_new > ALPHA_CHANGE_LIMIT
                rel_change = np.where(
                    monitored,
                    np.abs(counts_new - counts_old) / np.where(monitored, counts_new, 1.0),
                    0.0,
                )
                changed = int(np.sum(monitored & (rel_change > ALPHA_CHANGE)))
            else:
                # "joli" mode: compare normalized theta directly.
                # Monitor all active transcripts (theta > numerical floor).
                # ALPHA_CHANGE_LIMIT is NOT used here — it was designed for raw
                # counts in kallisto mode and incorrectly filters out most
                # transcripts when applied to normalized theta over 200k+ entries.
                changed = int(np.sum(
                    (theta_new > 1e-10) &
                    (np.abs(theta_new - theta) / np.maximum(theta_new, TOLERANCE)
                     > ALPHA_CHANGE)
                ))

            theta = theta_new

            # Save snapshot every snapshot_interval rounds
            if snapshots is not None and round_num % snapshot_interval == 0:
                snapshots.append((round_num, theta.copy()))

            # Log how many transcripts are being monitored (first 3 rounds only)
            if round_num < 3:
                if convergence_mode == "joli":
                    n_monitored = int((theta_new > 1e-10).sum())
                else:
                    n_monitored = int((theta_new * total_all_reads > ALPHA_CHANGE_LIMIT).sum())
                print(f"[JoliEM] Round {round_num + 1}: "
                      f"monitored={n_monitored}, changed={changed}")

            if changed == 0 and round_num >= min_rounds:
                converged = True
                print(f"[JoliEM] Converged at round {round_num + 1}")
                break

            # Progress checkpoint every 100 rounds
            if (round_num + 1) % 100 == 0:
                print(f"[JoliEM] Round {round_num + 1}: "
                      f"changed={changed}, "
                      f"nonzero_tx={int((theta > 0).sum())}")

        if not converged:
            print(f"[JoliEM] Reached max_em_rounds={max_em_rounds} without convergence.")

        # --- Zero small abundances ---
        if convergence_mode == "kallisto":
            # Threshold on raw expected counts. Use total_all_reads because theta
            # now normalises over all reads (single-tx + multi-tx).
            counts_temp = theta * total_all_reads
            n_zeroed = int(((counts_temp > 0) & (counts_temp < ALPHA_LIMIT / 10)).sum())
            theta[counts_temp < ALPHA_LIMIT / 10] = 0.0
        else:
            # "joli" mode: threshold on normalized theta (faster, for MAP/VI)
            n_zeroed = int(((theta > 0) & (theta < ALPHA_LIMIT / 10)).sum())
            theta[theta < ALPHA_LIMIT / 10] = 0.0
        if n_zeroed:
            print(f"[JoliEM] Zeroed {n_zeroed} transcripts below {ALPHA_LIMIT/10:.1e}")

        # --- finalRound (kallisto mode only) ---
        # One extra EM round after zeroing so reads redistribute away from zeroed
        # transcripts. Single-tx counts are included when the switch is on, keeping
        # finalRound consistent with the M-step used throughout the loop.
        if convergence_mode == "kallisto":
            n_final = self._em_step(theta)
            n_final_total = (n_final + self._single_tx_counts_per_tx) if self._em_include_single_tx else n_final
            n_final_total[~self._active_tx_mask] = 0.0
            total_final = n_final_total.sum()
            if total_final > 0:
                theta = n_final_total / total_final
                counts_final = theta * total_all_reads
                n_zeroed_final = int(
                    ((counts_final > 0) & (counts_final < ALPHA_LIMIT / 10)).sum()
                )
                theta[counts_final < ALPHA_LIMIT / 10] = 0.0
                print(f"[JoliEM] finalRound done. "
                      f"Additional transcripts zeroed: {n_zeroed_final}")

        # --- Build theta_unnorm: raw expected counts (not normalized, does not sum to 1) ---
        # Named theta_unnorm to distinguish from the Dirichlet alpha_prior.
        theta_unnorm = theta * total_all_reads
        # When em_include_single_tx=False, single-tx was excluded from the EM loop;
        # add raw counts back now so theta_unnorm is the complete expected count vector.
        if not self._em_include_single_tx and len(self._single_tx_ids) > 0:
            np.add.at(theta_unnorm, self._single_tx_ids, self._single_ec_counts)

        print(f"[JoliEM] Done. Rounds={round_num + 1}, "
              f"nonzero_transcripts={int((theta_unnorm > 0).sum())}")

        # Always append the final theta as the last snapshot
        if snapshots is not None:
            snapshots.append((round_num + 1, theta.copy()))

        return EMResult(theta_unnorm=theta_unnorm, n_rounds=round_num + 1, converged=converged,
                        snapshots=snapshots)

    def em_step(
        self,
        theta: np.ndarray,
        alpha_prior: np.ndarray = None,
        min_read_support: float = 0.0,
        convergence_mode: str   = "joli",
    ) -> tuple:
        """
        Run one combined E+M step without zeroing or finalRound.

        Used exclusively by MultiSampleJoliEM in em_wrapper (AT_code) mode,
        where 1 EM step per sample is interleaved with N GD steps per outer
        iteration, and the outer loop runs until EM convergence.

        Unlike run(), this method:
          - Does NOT zero small transcripts (no ALPHA_LIMIT thresholding).
          - Does NOT run finalRound.
        Those finalisation steps happen once after the outer loop ends, via a
        single call to run(max_em_rounds=1, init_theta=...) in write_results().

        Single-tx counts ARE included in the M-step (same as run()), so theta_new
        correctly reflects all-read evidence and the E-step probability for the
        next iteration is properly informed by single-tx support.

        Args:
            theta            : np.ndarray  -- current normalized abundances,
                                             shape [n_transcripts].
            alpha_prior      : np.ndarray  -- optional Dirichlet prior,
                                             shape [n_transcripts]. None = plain EM.
            min_read_support : float       -- Fix A threshold. Prior only applied
                                             when n[t] >= this. 0.0 = disabled.
            convergence_mode : str         -- "joli" (normalized theta change) or
                                             "kallisto" (raw expected count change).

        Returns:
            tuple(np.ndarray, int):
                theta_new : updated normalized abundances, shape [n_transcripts].
                n_changed : number of monitored transcripts still changing.
                            0 means this sample has converged for this step.
        """
        # E+M step: compute expected counts from multi-tx ECs only
        n = self._em_step(theta)

        # Add fixed single-tx counts when switch is on — mirrors the M-step in run()
        n_total = (n + self._single_tx_counts_per_tx) if self._em_include_single_tx else n

        # M-step: MAP posterior mean (with prior) or plain EM (without)
        if alpha_prior is not None:
            if min_read_support > 0.0:
                # Fix A: gate prior on read support
                has_support = n_total >= min_read_support
                numerator = np.where(has_support, n_total + alpha_prior, n_total)
            else:
                numerator = n_total + alpha_prior
        else:
            numerator = n_total

        # Zero out transcripts with no reads in this sample
        numerator[~self._active_tx_mask] = 0.0

        total = numerator.sum()
        theta_new = numerator / total if total > 0 else theta.copy()

        # Convergence check — uses total_all_reads when switch is on, total_multi_reads otherwise
        effective_total = self._total_all_reads if self._em_include_single_tx else self._total_multi_reads
        if convergence_mode == "kallisto":
            alpha_new = theta_new * effective_total
            alpha_old = theta     * effective_total
            monitored  = alpha_new > ALPHA_CHANGE_LIMIT
            rel_change = np.where(
                monitored,
                np.abs(alpha_new - alpha_old) / np.where(monitored, alpha_new, 1.0),
                0.0,
            )
            n_changed = int(np.sum(monitored & (rel_change > ALPHA_CHANGE)))
        else:
            n_changed = int(np.sum(
                (theta_new > 1e-10) &
                (np.abs(theta_new - theta) /
                 np.maximum(theta_new, TOLERANCE) > ALPHA_CHANGE)
            ))

        return theta_new, n_changed
