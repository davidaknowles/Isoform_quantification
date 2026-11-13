"""
joint.py
========
Joint EM over several libraries (short-read, long-read, any number of each)
that share molar transcript abundances.

Model
-----
Units k = 1..K (biological samples, cell types, metacells) have molar
abundances theta_k (T,), sum_t theta_kt = 1. Library l = 1..L observes the
mixture mu_l = sum_k pi_lk theta_k, where pi (L, K) holds known molecule
fractions (a one-hot row when the library comes from a single unit).
A read of library l falls in equivalence class e with probability

    P(e | l) = sum_{t in e} mu_lt c_let / sum_t mu_lt b_lt

where b_lt is the relative sampling rate of transcript t in library l
(effective length for short reads, 1 for full-length long reads) and c_let
the relative compatibility weight of t with e (1 by default; positional or
primer bias models enter here). With b = eff_len and c = 1 this is kallisto's
short-read likelihood written in molar units; with b = c = 1 it is the
long-read likelihood.

Update
------
E-step: z_lkt = sum_e n_le pi_lk theta_kt c_let / sum_{k',t' in e} pi_lk' theta_k't' c_let'.
The -N_l log B_l normaliser (B_l = sum_t mu_lt b_lt) is minorised by its
tangent, giving the MM update

    theta_kt = (Z_kt + a_kt) / (D_kt + lambda_k),
    Z_kt = sum_l z_lkt,   D_kt = sum_l N_l pi_lk b_lt / B_l,

with Dirichlet pseudo-counts a_kt (posterior-mean convention of JoliEM,
a = alpha) and lambda_k set so that theta_k sums to 1. For one long-read
library this is exactly JoliEM's MAP update (n + alpha) / (N + sum alpha).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .load_tcc import FlatECs, TCCData

TOLERANCE = np.finfo(np.float64).tiny


@dataclass
class Library:
    """
    One sequencing library as EC counts.

    Attributes:
        name     : str
        tcc      : TCCData           -- EC counts and EC -> transcript lists.
        b        : np.ndarray (T,)   -- sampling rate per transcript (eff_len for SR, 1 for LR).
        platform : str               -- "short" | "long" (informational).
        c        : list[np.ndarray] | None -- optional per-EC compatibility weights,
                   (any per-EC constant factor cancels in the EM)
                   c[e][i] for the i-th transcript of EC e; None means all ones.
        feat     : list[np.ndarray] | None -- optional per-EC integer features,
                   feat[e][i] (n_features,) for the i-th transcript of EC e, used by
                   bias models that recompute c (e.g. binned read-end positions).
    """
    name: str
    tcc: TCCData
    b: np.ndarray
    platform: str = "long"
    c: list = None
    feat: list = None


class _FlatLibrary:
    """Flat (EC, transcript) arrays of one library for vectorised E-steps."""

    def __init__(self, lib: Library, sort_ecs=False):
        counts = np.asarray(lib.tcc.ec_counts, dtype=np.float64)
        keep = np.flatnonzero(counts > 0)
        ecs = lib.tcc.ec_transcripts
        if isinstance(ecs, FlatECs):
            # flat input: gather the kept ECs' positions without per-EC Python objects
            sizes = np.diff(ecs.ptr)[keep]
            off = np.r_[0, np.cumsum(sizes)]
            src = np.repeat(ecs.ptr[keep] - off[:-1], sizes) + np.arange(off[-1])
            if len(keep) == len(counts):
                src = slice(None)
            self.tx = np.asarray(ecs.values[src], dtype=np.int64)
            self.c = np.ones(len(self.tx)) if lib.c is None else np.asarray(lib.c.values[src], dtype=np.float64)
            self.feat = None if lib.feat is None else np.asarray(lib.feat.values[src]).reshape(len(self.tx), -1)
        else:
            sizes = np.array([len(ecs[e]) for e in keep], dtype=np.int64)
            self.tx = np.concatenate([np.asarray(ecs[e], dtype=np.int64) for e in keep])
            if lib.c is None:
                self.c = np.ones(len(self.tx), dtype=np.float64)
            else:
                self.c = np.concatenate([np.asarray(lib.c[e], dtype=np.float64) for e in keep])
            self.feat = None if lib.feat is None else \
                np.concatenate([np.asarray(lib.feat[e]).reshape(sizes[i], -1) for i, e in enumerate(keep)])
        if sort_ecs and len(keep) > 1:
            # order ECs by their smallest transcript index: neighbouring ECs then touch nearby entries
            # of mu and g in the E-step (isoforms of a gene have adjacent indices)
            off = np.r_[0, np.cumsum(sizes)]
            order = np.argsort(np.minimum.reduceat(self.tx, off[:-1]), kind="stable")
            sizes = sizes[order]
            new_off = np.r_[0, np.cumsum(sizes)]
            src = np.repeat(off[:-1][order] - new_off[:-1], sizes) + np.arange(new_off[-1])
            self.tx, self.c = self.tx[src], self.c[src]
            if self.feat is not None:
                self.feat = self.feat[src]
            keep = keep[order]
        self.ec = np.repeat(np.arange(len(keep), dtype=np.int64), sizes)
        self.keep = keep                     # flat EC i is lib.tcc EC keep[i]
        self.ec_ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)   # EC i spans ec_ptr[i]:ec_ptr[i+1]
        self.counts = counts[keep]
        self.n_ec = len(keep)
        self.N = float(self.counts.sum())
        self.b = np.asarray(lib.b, dtype=np.float64)
        self.name = lib.name


@dataclass
class JointResult:
    theta: np.ndarray            # (K, T) molar abundances
    n_iter: int
    converged: bool
    loglik: list = field(default_factory=list)
    expected_reads: np.ndarray = None   # (L, T) expected reads per transcript and library


class JointEM:
    """
    MM/EM for molar abundances shared across libraries.

    Internally theta lives on the transcripts that appear in some library's
    ECs (compact space, `cols`), with per-library compact indices `txc` and
    rates `bc`; libraries keep global indices `tx` and rates `b` for bias
    models. run() takes and returns full-length arrays.

    Args:
        libraries : list[Library]
        pi        : np.ndarray (L, K) | None -- library x unit molecule fractions;
                    None means each library is its own unit (K = L).
        units     : list[int] | None -- shorthand for one-hot pi: units[l] = k.
        unit_names: list[str] | None
        sort_ecs  : bool -- store each library's ECs ordered by smallest transcript index (memory
                    locality in the E-step; fits are unchanged).
        tie       : (K,) int | None -- units with the same tie label share within-group fractions:
                    theta_ut = G_{u,g(t)} psi_{tie(u),t} with unit-specific group totals G (e.g. units
                    are cell type x platform, groups are genes, so platforms may differ in gene
                    expression but share isoform fractions). The M-step alternates closed-form
                    updates of G and psi (tie_iter rounds; one round already raises the MM surrogate).
        groups    : (T_full,) int -- group (gene) per transcript for tie; negative values get a
                    group of their own.
        pool      : None | "gene" | "transcript" | "design" -- partial pooling across units with the same tie
                    label instead of exact sharing. "gene": theta_ut = G_ug psi_jt with shared psi and
                    log G_ug ~ N(log H_jg + s_u, pool_tau^2); "transcript": log theta_ut ~ N(eta_jt + c_u,
                    pool_tau^2). Small pool_tau approaches a shared theta, large pool_tau separate units
                    (gene: shared psi with free gene totals). MAP by MM with Newton steps on the logs; the
                    profile (shared means and unit offsets) is maximised out, so the objective is a
                    function of theta.
                    "design": log theta_ut = c_u + sum_d X_ud B_dt + e_ut, e ~ N(0, pool_tau^2), with
                    B_dt ~ N(0, design_sd_d^2) (inf: flat) and free unit offsets c_u; all units are
                    active on every transcript seen in some library (transcripts without reads in a
                    unit are shrunk by its expected-read term), so the profile (c, B) is a closed-form
                    ridge solve shared by all transcripts. One-hot cell-type columns with flat priors
                    give transcript-level pooling; lineage, global and platform columns with finite sd
                    share information across cell types.
        design    : (K, D) design matrix of the units (pool="design").
        design_sd : (D,) prior sd of each column's effects (np.inf for unpenalised columns).
        base      : (K,) int, for pool="design_gene": -1 for base units (which carry transcript abundances
                    with the design prior; design has one row per base unit, in unit order), or the index of
                    the base unit that a scaled unit rescales per group: theta_ut = R_ug theta_bt (shared
                    within-group fractions, free group totals; e.g. SR units on their cell type's LR unit,
                    with genes as groups and unspliced targets as groups of their own).
        design_mstep : "alternate" | "exact" -- design M-step: design_inner rounds alternating the profile
                    and the per-entry update (default), or per-transcript Newton on the MM surrogate with
                    B profiled out (unit offsets held; fewer MM steps but ~100x costlier per step).
        backend   : "auto" | "numpy" | "rust" -- E-step implementation; "auto" uses the
                    compiled kernels (joli.native, rust/joli_kernels) when built.
        threads   : int | None -- threads for the compiled kernels.
    """

    def __init__(self, libraries, pi=None, units=None, unit_names=None, backend="auto", threads=None,
                 sort_ecs=True, tie=None, groups=None, tie_iter=1, pool=None, pool_tau=1.0, design=None,
                 design_sd=None, design_mstep="alternate", design_inner=3, base=None):
        from . import native
        if backend == "auto":
            backend = "rust" if native.available() else "numpy"
        if backend == "rust" and not native.available():
            raise RuntimeError("rust backend requested but rust/joli_kernels is not built")
        self.backend, self.threads = backend, threads
        self.libs = [_FlatLibrary(l, sort_ecs=sort_ecs) for l in libraries]
        self.L = len(self.libs)
        self.T_full = len(libraries[0].tcc.transcript_names)
        self.transcript_names = libraries[0].tcc.transcript_names
        for l in libraries[1:]:
            if l.tcc.transcript_names != self.transcript_names:
                raise ValueError(f"library {l.name} uses a different transcript index")
        if pi is None:
            if units is None:
                units = list(range(self.L))
            K = max(units) + 1
            pi = np.zeros((self.L, K))
            pi[np.arange(self.L), units] = 1.0
        self.pi = np.asarray(pi, dtype=np.float64)
        self.K = self.pi.shape[1]
        self.unit_names = unit_names or [f"unit{k}" for k in range(self.K)]
        # compact transcript space: transcripts in a nonzero EC of some library
        seen_any = np.zeros(self.T_full, dtype=bool)
        for fl in self.libs:
            seen_any[fl.tx] = True
        self.cols = np.flatnonzero(seen_any)
        self.T = len(self.cols)
        remap = np.full(self.T_full, -1, dtype=np.int64)
        remap[self.cols] = np.arange(self.T)
        for fl in self.libs:
            fl.txc = np.ascontiguousarray(remap[fl.tx])
            fl.bc = np.ascontiguousarray(fl.b[self.cols])
        # transcripts that appear in a nonzero EC of some library of unit k (compact)
        self.active = np.zeros((self.K, self.T), dtype=bool)
        for l, fl in enumerate(self.libs):
            seen = np.zeros(self.T, dtype=bool)
            seen[fl.txc] = True
            self.active[self.pi[l] > 0] |= seen
        self._activef = self.active.astype(np.float64)
        # one-hot pi (each library from one unit): E-step accumulation without (L, T) temporaries
        nz = self.pi > 0
        self._unit_of = np.argmax(self.pi, axis=1) if (nz.sum(axis=1) == 1).all() and \
            np.allclose(self.pi[nz], 1.0) else None
        # rows of G and the stacked rates: library order, or grouped by unit for one-hot pi so that
        # each unit's libraries are a contiguous slice (_unit_rows[k] = (start, stop))
        if self._unit_of is None:
            self._row = np.arange(self.L)
        else:
            order = np.argsort(self._unit_of, kind="stable")
            self._row = np.empty(self.L, dtype=np.int64)
            self._row[order] = np.arange(self.L)
            bounds = np.searchsorted(self._unit_of[order], np.arange(self.K + 1))
            self._unit_rows = [(int(bounds[k]), int(bounds[k + 1])) for k in range(self.K)]
        self._Bmat = None                     # stacked compact rates (L, T), rebuilt after set_bias
        self._G = np.empty((self.L, self.T))  # per-library gradients g_l (L, T), row self._row[l]
        self._N = np.array([fl.N for fl in self.libs])
        self._tie = None
        self._pooling = None
        if pool == "design_gene":
            base = np.asarray(base, dtype=np.int64)
            bunits = np.flatnonzero(base < 0)
            X = np.asarray(design, dtype=np.float64)
            sd = np.asarray(design_sd, dtype=np.float64)
            if len(base) != self.K or X.shape[0] != len(bunits) or sd.shape != (X.shape[1],):
                raise ValueError("design_gene needs base per unit and one design row per base unit")
            if np.any(base[base >= 0] < 0) or np.any(base[base[base >= 0]] >= 0):
                raise ValueError("scaled units must point at base units")
            tau2 = float(pool_tau) ** 2
            lam = np.where(np.isfinite(sd), 1.0 / np.maximum(sd, 1e-300) ** 2, 0.0)
            proj = np.linalg.pinv(X.T @ X / tau2 + np.diag(lam)) @ X.T / tau2
            g = np.asarray(groups, dtype=np.int64)[self.cols].copy()
            lone = g < 0
            g[lone] = g.max() + 1 + np.arange(lone.sum())
            gc, _ = pd.factorize(g)
            self._pooling = {"level": "design_gene", "tau2": tau2, "state": None, "X": X, "lam": lam,
                             "proj": proj, "base": base, "bunits": bunits, "gc": gc.astype(np.int64),
                             "n_g": int(gc.max()) + 1, "inner": int(design_inner)}
            self.active[:] = self.active.any(axis=0)
            self._activef = self.active.astype(np.float64)
        elif pool == "design":
            X = np.asarray(design, dtype=np.float64)
            sd = np.asarray(design_sd, dtype=np.float64)
            if X.shape[0] != self.K or sd.shape != (X.shape[1],):
                raise ValueError("design needs one row per unit and design_sd one value per column")
            tau2 = float(pool_tau) ** 2
            lam = np.where(np.isfinite(sd), 1.0 / np.maximum(sd, 1e-300) ** 2, 0.0)
            # ridge solve on unit-centred log abundances; pinv for collinear unpenalised columns
            proj = np.linalg.pinv(X.T @ X / tau2 + np.diag(lam)) @ X.T / tau2          # (D, K)
            # profiled prior given unit offsets: -1/2 sum_t (x_t - c)' Q (x_t - c), Q = (I - X proj) / tau^2
            Q = (np.eye(self.K) - X @ proj) / tau2
            Q = 0.5 * (Q + Q.T)
            self._pooling = {"level": "design", "tau2": tau2, "state": None, "X": X, "lam": lam,
                             "proj": proj, "Q": Q, "mstep": design_mstep, "inner": int(design_inner)}
            self.active[:] = self.active.any(axis=0)
            self._activef = self.active.astype(np.float64)
        elif pool is not None:
            if tie is None or (pool == "gene" and groups is None):
                raise ValueError("pool needs tie (and groups for gene-level pooling)")
            self._pooling = {"level": pool, "tau2": float(pool_tau) ** 2, "state": None}
            if groups is None:
                groups = np.arange(self.T_full)
        if tie is not None:
            g = np.asarray(groups, dtype=np.int64)[self.cols].copy()
            lone = g < 0
            g[lone] = g.max() + 1 + np.arange(lone.sum())
            gc, _ = pd.factorize(g)
            tie = np.asarray(tie, dtype=np.int64)
            if len(tie) != self.K:
                raise ValueError("tie needs one label per unit")
            codes, self._tie_labels = pd.factorize(tie)
            self._tie = (codes, gc.astype(np.int64), int(gc.max()) + 1, int(tie_iter))
            # units sharing fractions are active on the union of their transcripts
            for j in np.unique(self._tie[0]):
                us = self._tie[0] == j
                self.active[us] = self.active[us].any(axis=0)
            self._activef = self.active.astype(np.float64)

    def _compact(self, x):
        """Full-length (..., T_full) array to compact (..., T)."""
        return np.asarray(x)[..., self.cols]

    def _expand(self, x):
        """Compact (..., T) array to full length (..., T_full), zeros elsewhere."""
        out = np.zeros(x.shape[:-1] + (self.T_full,))
        out[..., self.cols] = x
        return out

    def _library_estep(self, l, mu, threads, out=None):
        """(g, sum_e n_e log denom_e) for library l at compact molar mixture mu; g written to out if given."""
        fl = self.libs[l]
        if self.backend == "rust":
            from . import native
            if threads > 1 and getattr(self, "split_kernel", "blocks") == "gather":
                if getattr(fl, "gather", None) is None:
                    fl.gather = native.gather_index(fl.txc, fl.ec, self.T)
                if getattr(fl, "c_sorted", None) is None:      # reset by set_bias
                    fl.c_sorted = np.ascontiguousarray(fl.c[fl.gather[1]])
                return native.estep(fl.ec_ptr, fl.txc, fl.c, fl.counts, mu, threads, fl.gather, fl.c_sorted, out=out)
            return native.estep(fl.ec_ptr, fl.txc, fl.c, fl.counts, mu, threads, out=out)
        w = mu[fl.txc] * fl.c                                   # per position
        denom = np.bincount(fl.ec, weights=w, minlength=fl.n_ec)
        ok = denom > TOLERANCE
        ratio = np.where(ok, fl.counts / np.where(ok, denom, 1.0), 0.0)
        r_pos = fl.c * ratio[fl.ec]                             # n_e c_et / denom_e
        g = np.bincount(fl.txc, weights=r_pos, minlength=self.T)  # sum_e n_e c_et / denom_e
        if out is not None:
            out[:] = g
            g = out
        return g, float(fl.counts[ok] @ np.log(denom[ok]))

    def _library_esteps(self, mu_of):
        """
        Per-library E-steps into self._G; returns the log-likelihood terms. With the rust backend and
        several threads, libraries holding more than big_fraction / threads of all positions run one at a
        time split over all threads (EC blocks, per-thread accumulators; split_kernel = "gather" uses the
        two-pass kernel instead), and the rest concurrently in a thread pool, largest first. On real
        Kinnex + Parse libraries (50M positions, 16 threads) big_fraction 2 was fastest.
        """
        threads = 1
        if self.backend == "rust":
            from . import native
            threads = self.threads or native.default_threads()
        G, sizes = self._G, [len(fl.tx) for fl in self.libs]
        total = sum(sizes)
        lls = [0.0] * self.L
        if threads == 1 or total < 1_000_000:
            for l in range(self.L):
                lls[l] = self._library_estep(l, mu_of(l), min(threads, max(1, sizes[l] // 5_000_000)),
                                             out=G[self._row[l]])[1]
            return lls
        frac = getattr(self, "big_fraction", 2.0)       # libraries above frac / threads of all positions
        big = [l for l in range(self.L) if sizes[l] > frac * total / threads and sizes[l] >= 2_000_000]
        for l in big:
            lls[l] = self._library_estep(l, mu_of(l), threads, out=G[self._row[l]])[1]
        rest = sorted(set(range(self.L)) - set(big), key=lambda l: -sizes[l])
        if rest:
            if getattr(self, "_pool", None) is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(max_workers=threads)
            run = lambda l: self._library_estep(l, mu_of(l), 1, out=G[self._row[l]])[1]
            for l, ll in zip(rest, self._pool.map(run, rest)):
                lls[l] = ll
        return lls

    def _estep(self, theta, expected_reads=False):
        """
        For compact theta (K, T): Z (K, T), D (K, T), loglik, and expected reads
        per library (L, T, compact) if requested (else None). With the rust
        backend and several threads, libraries run concurrently (ctypes releases
        the GIL; see _library_esteps). Accumulation over libraries uses two matrix
        products, Z = theta * (pi^T G) and D = (pi * N / B)^T Bmat, or per-unit
        slice sums for one-hot pi (_estep_onehot).
        """
        if self._Bmat is None:
            self._Bmat = np.empty((self.L, self.T))
            for l, fl in enumerate(self.libs):
                self._Bmat[self._row[l]] = fl.bc
        if self._unit_of is not None:
            return self._estep_onehot(theta, expected_reads)
        mus = self.pi @ theta                                    # (L, T)
        G = self._G
        lls = self._library_esteps(lambda l: mus[l])
        B = np.einsum("lt,lt->l", mus, self._Bmat)
        loglik = float(np.sum(lls) - self._N @ np.log(B))
        Z = theta * (self.pi.T @ G)
        D = (self.pi * (self._N / B)[:, None]).T @ self._Bmat
        R = mus * G if expected_reads else None
        return Z, D, loglik, R

    def _estep_onehot(self, theta, expected_reads):
        """_estep for one-hot pi: mu_l is a row of theta, and Z, D are sums over each unit's libraries."""
        uo = self._unit_of
        G = self._G
        lls = self._library_esteps(lambda l: theta[uo[l]])
        Bm, rows = self._Bmat, self._row
        Bs = np.empty(self.L)                        # B in row order
        Z = np.empty_like(theta)
        D = np.empty_like(theta)
        for k, (a, b) in enumerate(self._unit_rows):
            Bs[a:b] = Bm[a:b] @ theta[k]
        B = Bs[rows]
        loglik = float(np.sum(lls) - self._N @ np.log(B))
        ws = np.empty(self.L)
        ws[rows] = self._N / B
        for k, (a, b) in enumerate(self._unit_rows):
            if b > a:
                np.multiply(theta[k], np.add.reduce(G[a:b], axis=0), out=Z[k])
                D[k] = ws[a:b] @ Bm[a:b]
            else:
                Z[k] = 0.0
                D[k] = 0.0
        R = theta[uo] * G[rows] if expected_reads else None
        return Z, D, loglik, R

    def position_counts(self, theta, l):
        """
        Expected reads per (EC, transcript) position of library l,
        z_lp = n_e mu_t c_et / sum_{t' in e} mu_t' c_et', as a flat array
        aligned with the library's flat EC layout (for M-steps of bias models).
        theta is full length (K, T_full), as returned by run().
        """
        fl = self.libs[l]
        mu = self.pi[l] @ self._compact(theta)
        if self.backend == "rust":
            from . import native
            return native.position_counts(fl.ec_ptr, fl.txc, fl.c, fl.counts, mu,
                                          min(self.threads or native.default_threads(), max(1, len(fl.tx) // 1_000_000)))
        w = mu[fl.txc] * fl.c
        denom = np.bincount(fl.ec, weights=w, minlength=fl.n_ec)
        return np.where(denom[fl.ec] > TOLERANCE, fl.counts[fl.ec] * w / np.maximum(denom[fl.ec], TOLERANCE), 0.0)

    def library_loglik(self, theta, eps=0.0):
        """
        Log-likelihood of each library at fixed abundances (e.g. held-out libraries scored with
        abundances fitted elsewhere): ll_l = sum_e n_e log(sum_t mu_t c_et) - N_l log(sum_t mu_t b_t),
        mu_l = pi_l theta. theta is full length (K, T_full); eps mixes in eps of a uniform
        distribution over all targets so ECs whose transcripts have zero abundance stay finite.
        Returns (ll (L,), n_zero (L,)): n_zero counts reads in ECs of zero probability before smoothing.
        """
        th = np.asarray(theta, dtype=np.float64)
        th_s = (1.0 - eps) * th + eps / th.shape[1]
        th, th_s = self._compact(th), self._compact(th_s)
        ll, n_zero = np.zeros(self.L), np.zeros(self.L)
        for l, fl in enumerate(self.libs):
            out = []
            for t in (th, th_s):
                mu = self.pi[l] @ t
                denom = np.bincount(fl.ec, weights=mu[fl.txc] * fl.c, minlength=fl.n_ec)
                out.append((denom, mu))
            (d0, _), (d, mu) = out
            n_zero[l] = fl.counts[d0 <= TOLERANCE].sum()
            ok = d > TOLERANCE
            ll[l] = fl.counts[ok] @ np.log(d[ok]) - fl.N * np.log(mu @ fl.bc) \
                + (-np.inf if (~ok & (fl.counts > 0)).any() else 0.0)
        return ll, n_zero

    def ec_logprob(self, theta, groups=None):
        """
        Per-read log-probability of each EC of each library at fixed abundances (no smoothing):
        log sum_{t in e} mu_t c_et - log sum_t mu_t b_t, or with groups (T_full,) the probability
        given the group, - log sum_{t in g(e)} mu_t b_t, NaN for ECs spanning several groups.
        -inf marks ECs of zero probability. Returns a list over libraries of (n_ec,) arrays aligned
        with each library's flat EC order (fl.counts), so scores of different fits of the same
        libraries can be restricted to common ECs.
        """
        th = self._compact(np.asarray(theta, dtype=np.float64))
        gc = None if groups is None else np.asarray(groups)[self.cols]
        out = []
        for l, fl in enumerate(self.libs):
            mu = self.pi[l] @ th
            d = np.bincount(fl.ec, weights=mu[fl.txc] * fl.c, minlength=fl.n_ec)
            with np.errstate(divide="ignore"):
                ld = np.log(d)
            if gc is None:
                out.append(ld - np.log(mu @ fl.bc))
                continue
            g = gc[fl.txc]
            gmin = np.minimum.reduceat(g, fl.ec_ptr[:-1])
            gmax = np.maximum.reduceat(g, fl.ec_ptr[:-1])
            Bg = np.bincount(np.maximum(gc, 0), weights=np.where(gc >= 0, mu * fl.bc, 0.0), minlength=int(gc.max()) + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = ld - np.log(Bg[np.maximum(gmin, 0)])
            out.append(np.where((gmin == gmax) & (gmin >= 0), lp, np.nan))
        return out

    def library_loglik_within(self, theta, groups, eps=0.0):
        """
        Within-group log-likelihood of each library at fixed abundances: over ECs whose
        transcripts all belong to one group g (e.g. a gene),
        ll_l = sum_e n_e [log sum_{t in e} mu_t c_et - log sum_{t in g(e)} mu_t b_t],
        the probability of the read given its group. Group-level expression (which may differ
        between donors or platforms) then does not enter, only the split within groups.
        groups : (T_full,) integer group code per transcript (negative: no group).
        Returns (ll (L,), reads used (L,), reads with zero probability before smoothing (L,)).
        """
        th = np.asarray(theta, dtype=np.float64)
        th_s = self._compact((1.0 - eps) * th + eps / th.shape[1])
        th0 = self._compact(th)
        gc = np.asarray(groups)[self.cols]
        G = int(gc.max()) + 1
        ll, used, n_zero = np.zeros(self.L), np.zeros(self.L), np.zeros(self.L)
        for l, fl in enumerate(self.libs):
            g = gc[fl.txc]
            gmin = np.minimum.reduceat(g, fl.ec_ptr[:-1])
            gmax = np.maximum.reduceat(g, fl.ec_ptr[:-1])
            one = (gmin == gmax) & (gmin >= 0)
            for t, which in ((th0, "raw"), (th_s, "smooth")):
                mu = self.pi[l] @ t
                d = np.bincount(fl.ec, weights=mu[fl.txc] * fl.c, minlength=fl.n_ec)
                Bg = np.bincount(np.maximum(gc, 0), weights=np.where(gc >= 0, mu * fl.bc, 0.0), minlength=G)
                if which == "raw":
                    n_zero[l] = fl.counts[one & (d <= TOLERANCE)].sum()
                    continue
                ok = one & (d > TOLERANCE) & (Bg[np.maximum(gmin, 0)] > TOLERANCE)
                ll[l] = fl.counts[ok] @ (np.log(d[ok]) - np.log(Bg[gmin[ok]]))
                used[l] = fl.counts[ok].sum()
        return ll, used, n_zero

    def _design_profile(self, x):
        """(c, B, m): unit offsets c_u = mean_t x_ut, ridge effects B (D, T) of the centred x, prior mean m."""
        P = self._pooling
        c = x.mean(axis=1)
        B = P["proj"] @ (x - c[:, None])
        return c, B, c[:, None] + P["X"] @ B

    def _design_logprior(self, x, c, B, m):
        P = self._pooling
        r = x - m
        return float(-0.5 * (r * r).sum() / P["tau2"] - 0.5 * (P["lam"] @ (B * B).sum(axis=1)))

    def _pooled_design_mstep(self, theta, Z, D):
        """
        MAP-MM step for the design prior: profile (c, B) maximised at the input (closed form), then each
        x_ut = log theta_ut set to the exact maximiser of Z x - D e^x - (x - m)^2 / (2 tau^2).
        """
        P = self._pooling
        x = np.log(np.maximum(theta, 1e-300))
        c, B, m = self._design_profile(x)
        lp = self._design_logprior(x, c, B, m)
        if P["mstep"] == "exact":
            x = _design_newton(x, Z, D, c, P["Q"])
        else:
            for i in range(P["inner"]):
                if i:
                    c, B, m = self._design_profile(x)
                x = _newton_lognormal(x, Z, D, m, P["tau2"], True)
        P["state"] = self._design_profile(x)[:2]
        new = np.exp(x - x.max(axis=1, keepdims=True))
        new /= new.sum(axis=1, keepdims=True)
        return new, lp

    def _design_gene_mstep(self, theta, Z, D):
        """
        MAP-MM step for pool="design_gene": base units carry x = log theta with the design prior; each
        scaled unit s on base b is theta_s = R_sg theta_b. The profile (c, B) of the base units is maximised
        at the input; then design_inner rounds of R_sg = Z_sg / sum_{t in g} D_st e^{x_bt} (closed form)
        and, per base entry, the exact maximiser of (Z_b + sum_s Z_s) x - (D_b + sum_s R_s D_s) e^x
        - (x - m)^2 / (2 tau^2); each step raises the surrogate.
        """
        P = self._pooling
        base, bunits, gc, n_g = P["base"], P["bunits"], P["gc"], P["n_g"]
        agg = lambda v: np.bincount(gc, weights=v, minlength=n_g)
        xb = np.log(np.maximum(theta[bunits], 1e-300))
        c = xb.mean(axis=1)
        B = P["proj"] @ (xb - c[:, None])
        m = c[:, None] + P["X"] @ B
        r = xb - m
        lp = float(-0.5 * (r * r).sum() / P["tau2"] - 0.5 * (P["lam"] @ (B * B).sum(axis=1)))
        row_of = {int(u): i for i, u in enumerate(bunits)}
        scaled = [(int(s), row_of[int(base[s])]) for s in np.flatnonzero(base >= 0)]
        Zb, Db = Z[bunits].copy(), D[bunits]
        for s, i in scaled:
            Zb[i] += Z[s]
        R = {}
        for it in range(P["inner"]):
            if it:
                c = xb.mean(axis=1)
                m = c[:, None] + P["X"] @ (P["proj"] @ (xb - c[:, None]))
            Dn = Db.copy()
            for s, i in scaled:
                den = agg(D[s] * np.exp(xb[i]))
                R[s] = np.divide(agg(Z[s]), den, out=np.zeros(n_g), where=den > 0)
                R[s] += 1e-9 * R[s].sum() / n_g            # keep scaled units positive in groups without reads
                Dn[i] += R[s][gc] * D[s]
            xb = _newton_lognormal(xb, Zb, Dn, m, P["tau2"], True)
        c = xb.mean(axis=1)
        P["state"] = (c, P["proj"] @ (xb - c[:, None]))
        new = np.zeros_like(theta)
        eb = np.exp(xb - xb.max(axis=1, keepdims=True))
        new[bunits] = eb
        for s, i in scaled:
            den = agg(D[s] * np.exp(xb[i]))
            Rs = np.divide(agg(Z[s]), den, out=np.zeros(n_g), where=den > 0)
            Rs += 1e-9 * Rs.sum() / n_g
            new[s] = Rs[gc] * eb[i]
        new *= self._activef
        new /= new.sum(axis=1, keepdims=True)
        return new, lp

    def design_profile(self, rows):
        """
        Molar fractions exp(r . B) normalised over transcripts for design rows r (n, D) (e.g. a cell
        type's intercept + lineage + cell-type columns without platform), from the last M-step.
        Returns (n, T_full).
        """
        if self._pooling is None or self._pooling["level"] not in ("design", "design_gene") or self._pooling["state"] is None:
            raise ValueError("needs a fitted design-pooled model")
        B = self._pooling["state"][1]
        e = np.atleast_2d(np.asarray(rows, dtype=np.float64)) @ B
        out = np.exp(e - e.max(axis=1, keepdims=True))
        out /= out.sum(axis=1, keepdims=True)
        return self._expand(out)

    def pooled_profile(self):
        """
        Shared profile of each tie group from the last transcript-level pooled M-step,
        exp(eta_jt) normalised over transcripts: the consensus molar fractions of the units in
        group j (e.g. a cell type across platforms). Returns (tie labels, (n_tie, T_full) array).
        """
        if self._pooling is None or self._pooling["level"] != "transcript" or self._pooling["state"] is None:
            raise ValueError("needs a fitted transcript-level pooled model")
        h = self._pooling["state"][0]
        j, t = np.divmod(self._act_gkey, self.T)
        n_tie = len(self._tie_labels)
        out = np.zeros((n_tie, self.T))
        mx = np.full(n_tie, -np.inf)
        np.maximum.at(mx, j, h)
        out[j, t] = np.exp(h - mx[j])
        out /= out.sum(axis=1, keepdims=True)
        return list(self._tie_labels), self._expand(out)

    def set_bias(self, l, c=None, b=None):
        """Replace library l's flat compatibility weights c (aligned with its flat layout) and/or full-length rates b."""
        if c is not None:
            self.libs[l].c = np.ascontiguousarray(c, dtype=np.float64)
            self.libs[l].c_sorted = None
        if b is not None:
            self.libs[l].b = np.ascontiguousarray(b, dtype=np.float64)
            self.libs[l].bc = np.ascontiguousarray(self.libs[l].b[self.cols])
            self._Bmat = None

    def _update(self, theta, a):
        """One MM step on compact theta. Returns (new theta, log posterior at the input theta)."""
        Z, D, ll, _ = self._estep(theta)
        if self._pooling is not None and self._pooling["level"] == "design":
            if a is not None:
                raise NotImplementedError("design pooling with a Dirichlet prior")
            new, lp = self._pooled_design_mstep(theta, Z, D)
            return new, ll + lp
        if self._pooling is not None and self._pooling["level"] == "design_gene":
            if a is not None:
                raise NotImplementedError("design pooling with a Dirichlet prior")
            new, lp = self._design_gene_mstep(theta, Z, D)
            return new, ll + lp
        if self._tie is not None:
            if a is not None:
                raise NotImplementedError("tied fractions with a Dirichlet prior")
            if self._pooling is not None and self._pooling["level"] == "transcript":
                new, lp = self._pooled_transcript_mstep(theta, Z, D)
                return new, ll + lp
            if self._pooling is not None:
                new, lp = self._tied_mstep(theta, Z, D, pooled=True)
                return new, ll + lp
            return self._tied_mstep(theta, Z, D), ll
        if a is None:
            num = Z
        else:
            num = Z + a                                        # a is zero off the active set
            pos = theta > 0
            ll += float((a[pos] * np.log(theta[pos])).sum())
        # lambda_k from sum_t theta_kt = 1 at the fixed point: sum(num) - sum(theta_old * D);
        # inactive entries have num = 0 (theta = 0 there and a masked), so stay 0
        lam = num.sum(axis=1, keepdims=True) - np.einsum("kt,kt->k", theta, D)[:, None]
        D += lam
        np.maximum(D, TOLERANCE, out=D)
        new = np.divide(num, D, out=num if a is None else None)
        new /= new.sum(axis=1, keepdims=True)
        return new, ll

    def _tied_mstep(self, theta, Z, D, pooled=False):
        """
        MM step for theta_ut = G_ug psi_jt (j = tie(u), g = group(t)): coordinate ascent on
        Q = sum_ut Z_ut log theta_ut - D_ut theta_ut (the likelihood is invariant to each unit's
        scale, so no sum constraint), G_ug = Z_ug / sum_{t in g} D_ut psi_jt and
        psi_jt = sum_{u in j} Z_ut / sum_{u in j} D_ut G_ug, with psi normalised within groups. G gets a
        floor of 1e-9 of the unit's mean group total, so units keep psi in groups without their own reads.
        """
        tie, gc, n_g, n_iter = self._tie
        if getattr(self, "_tie_mats", None) is None:
            import scipy.sparse as sp
            A = sp.csr_matrix((np.ones(self.T), (np.arange(self.T), gc)), shape=(self.T, n_g))  # transcript -> group
            M = np.zeros((int(tie.max()) + 1, self.K))
            M[tie, np.arange(self.K)] = 1.0                                                      # tie group x unit
            self._tie_mats = (A, M)
        A, M = self._tie_mats
        agg = lambda X: np.stack([np.bincount(gc, weights=x, minlength=n_g) for x in X])   # (rows, T) -> (rows, n_g)
        div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
        Gu = agg(theta)
        Zg = agg(Z)
        psi = div(M @ theta, M @ Gu[:, gc])
        Zj = M @ Z
        lp = 0.0
        if pooled:
            # log-normal prior on gene totals around a shared profile: state (h (n_tie, n_g), s (K,))
            gact = agg(self._activef) > 0                                  # groups active per unit
            y = np.log(np.maximum(Gu, 1e-300))
            # profile (h, s) maximised at the input, so the objective depends on theta alone
            st = self._lognormal_profile(y, gact, tie, M)
            lp = self._lognormal_logprior(y, st, gact, tie)
        for _ in range(n_iter):
            if pooled:
                Dg = agg(D * psi[tie])
                y = _newton_lognormal(y, Zg, Dg, st[0][tie] + st[1][:, None], self._pooling["tau2"], gact)
                st = self._lognormal_profile(y, gact, tie, M)
                Gu = np.where(gact, np.exp(y), 0.0)
            else:
                Gu = div(Zg, agg(D * psi[tie]))
                # tiny floor so a unit keeps the shared fractions psi in groups it has no reads for
                Gu += 1e-9 * Gu.sum(axis=1, keepdims=True) / n_g
            psi = div(Zj, M @ (D * Gu[:, gc]))
            tot = agg(psi)                                                # gauge: psi sums to 1 in groups
            psi = div(psi, tot[:, gc])
            Gu *= tot[tie]
            if pooled:
                y = np.log(np.maximum(Gu, 1e-300))
        new = Gu[:, gc] * psi[tie]
        new *= self._activef
        new /= new.sum(axis=1, keepdims=True)
        if pooled:
            self._pooling["state"] = st
            return new, lp
        return new

    def _lognormal_profile(self, y, act, tie, M):
        """
        Profile maximising the log prior given y: units of a tie group share their active set, so
        with the offsets s centred within each group, h_j = mean_{u in j} y_u and
        s_u = mean_{active t} (y_ut - h_jt) (exact, one pass).
        """
        a = act.astype(np.float64)
        ya = np.where(act, y, 0.0)
        cnt = M @ a
        h = np.divide(M @ ya, cnt, out=np.zeros((M.shape[0], y.shape[1])), where=cnt > 0)
        r = np.where(act, y - h[tie], 0.0)
        s = r.sum(axis=1) / np.maximum(a.sum(axis=1), 1.0)
        return h, s

    def _lognormal_logprior(self, y, st, act, tie):
        r = np.where(act, y - st[0][tie] - st[1][:, None], 0.0)
        return float(-0.5 * (r * r).sum() / self._pooling["tau2"])

    def _pooled_transcript_mstep(self, theta, Z, D):
        """
        MAP-MM step for log theta_ut ~ N(eta_jt + c_u, tau^2) (j = tie(u)): the profile (eta, c) is
        maximised at the input (so the objective depends on theta alone), then x = log theta is set
        to the exact maximiser of Q = sum Z x - D e^x + log prior (_newton_lognormal), all on the
        flat vector of active (unit, transcript) entries. Units of a tie group share their active
        set, so transcripts without reads in a unit get abundance from the other units of its group
        through eta.
        """
        tie = self._tie[0]
        if getattr(self, "_act_flat", None) is None:
            u, t = np.nonzero(self.active)
            key = tie[u] * self.T + t                               # (tie group, transcript)
            gkey, ginv = np.unique(key, return_inverse=True)
            self._act_gkey = gkey
            self._act_flat = (u, u * self.T + t, ginv, len(gkey), np.bincount(ginv), np.bincount(u, minlength=self.K))
        u, flat, ginv, n_gk, gcount, ucount = self._act_flat
        x = np.log(np.maximum(np.take(theta, flat), 1e-300))

        def profile(x):
            h = np.bincount(ginv, weights=x, minlength=n_gk) / gcount     # eta_jt (mean over units of j)
            c = np.bincount(u, weights=x - h[ginv], minlength=self.K) / np.maximum(ucount, 1)
            return h, c

        h, c = profile(x)
        r = x - h[ginv] - c[u]
        lp = float(-0.5 * (r @ r) / self._pooling["tau2"])
        x = _newton_lognormal(x, np.take(Z, flat), np.take(D, flat), h[ginv] + c[u], self._pooling["tau2"], True)
        self._pooling["state"] = profile(x)
        new = np.zeros_like(theta)
        starts = np.searchsorted(u, np.arange(self.K))             # entries are ordered by unit
        mx = np.full(self.K, -np.inf)
        has = ucount > 0
        mx[has] = np.maximum.reduceat(x, starts[has])
        np.put(new, flat, np.exp(x - mx[u]))
        new /= new.sum(axis=1, keepdims=True)
        return new, lp

    def _anderson(self, theta, a, max_iter, min_iter, rel_tol, theta_floor, ll_rel_tol, verbose, m=5, reg=1e-10,
                  ll_window=10):
        """
        Anderson acceleration (type II, memory m) of the MM map F in x = log theta on active entries:
        with residuals f_k = F(x_k) - x_k, x_{k+1} = F(x_k) - dG gamma, gamma minimising
        |f_k - dF gamma| (dF, dG differences of the last m residuals and images). One MM step per
        iterate; a candidate whose log posterior (from the next MM step) falls below the last accepted
        one is replaced by the plain MM image F(x_k), which cannot decrease it, and the history restarts.
        Returns (theta, log-posterior history, converged, MM steps).
        """
        act = self._activef > 0
        lx = lambda th: np.log(np.maximum(th[act], 1e-300))
        def from_x(x):
            th = np.zeros_like(theta)
            v = np.full(theta.shape, -np.inf)
            v[act] = x
            th = np.exp(v - v.max(axis=1, keepdims=True))
            return th / th.sum(axis=1, keepdims=True)
        history, converged, n_steps = [], False, 0
        Fth, ll = self._update(theta, a)                 # image of theta and log posterior at theta
        n_steps += 1
        x, g = lx(theta), lx(Fth)
        Xs, Gs = [x], [g]
        while n_steps < max_iter:
            f = Gs[-1] - Xs[-1]
            if len(Xs) > 1:
                dF = np.stack([(Gs[i + 1] - Xs[i + 1]) - (Gs[i] - Xs[i]) for i in range(len(Xs) - 1)], axis=1)
                dG = np.stack([Gs[i + 1] - Gs[i] for i in range(len(Xs) - 1)], axis=1)
                A = dF.T @ dF
                gam = np.linalg.solve(A + reg * np.trace(A) * np.eye(len(A)) + 1e-300 * np.eye(len(A)), dF.T @ f)
                xn = Gs[-1] - dG @ gam
                if not np.all(np.isfinite(xn)):            # degenerate least squares: plain MM image
                    xn = Gs[-1]
            else:
                xn = Gs[-1]
            cand = from_x(xn)
            Fc, llc = self._update(cand, a)
            n_steps += 1
            if not np.isfinite(llc) or llc < ll - 1e-12 * abs(ll):
                # reject: take the plain MM image of the last accepted point and restart the history
                cand = from_x(Gs[-1])
                Fc, llc = self._update(cand, a)
                n_steps += 1
                Xs, Gs = [], []
            history.append(ll)
            # stall: the gain over the last ll_window iterates is small (single steps can be tiny on flat tails)
            ll_stall = (ll_rel_tol is not None and len(history) >= ll_window
                        and 0.0 <= llc - history[-ll_window] < ll_window * ll_rel_tol * abs(llc))
            prev = from_x(Xs[-1]) if Xs else theta
            diff = np.abs(cand - prev)
            changed = int(np.count_nonzero((cand > theta_floor) & (diff > rel_tol * cand)))
            theta, ll = cand, llc
            Xs.append(lx(cand)); Gs.append(lx(Fc))
            Xs, Gs = Xs[-(m + 1):], Gs[-(m + 1):]
            if verbose and (len(history) <= 3 or len(history) % 50 == 0):
                print(f"[JointEM/anderson] iterate {len(history)} ({n_steps} MM steps): logpost={ll:.6e} changed={changed}")
            if n_steps >= min_iter and (changed == 0 or ll_stall):
                converged = True
                break
        return theta, history, converged, n_steps

    def run(self, prior=None, init_theta=None, max_iter=5000, min_iter=20,
            rel_tol=1e-2, theta_floor=1e-8, accelerate="auto", zero_below=0.01,
            ll_rel_tol=None, verbose=False, ll_window=None) -> JointResult:
        """
        Iterate until no monitored theta (> theta_floor) changes by more than
        rel_tol relatively between successive iterates (kallisto's rule; its
        0.01-read floor corresponds to theta_floor).

        With accelerate=True (or "squarem") each iterate is a SQUAREM step (Varadhan and
        Roland 2008, scheme S3): two MM steps, an extrapolation, and one
        stabilising MM step, falling back to the plain second MM step when the
        extrapolated point has lower log posterior. accelerate="anderson" uses
        Anderson acceleration of the MM map instead (_anderson). "auto" (default) uses
        Anderson for transcript-level and design pooling, whose units are coupled through the
        prior (about 18x fewer MM steps in simulation), and SQUAREM otherwise. n_iter counts MM steps.

        After convergence, transcripts of unit k whose expected read count over
        all libraries is below zero_below are set to zero and one more MM step
        is run (kallisto's finalRound). MLEs on the simplex boundary otherwise
        stay at small positive values whose number depends on the iteration count.

        Args:
            prior      : np.ndarray (T,) or (K, T) | None -- Dirichlet pseudo-counts a (full length).
            init_theta : np.ndarray (K, T) | None -- warm start (full length); uniform over active otherwise.
            zero_below : float | None -- expected-read threshold for final zeroing.
            ll_rel_tol : float | None -- also stop when the last ll_window iterates together raise the log
                         posterior by less than ll_window * ll_rel_tol * |log posterior| (weakly identified
                         fits, where theta drifts along nearly flat directions long after the likelihood has
                         settled).
            ll_window  : int | None -- iterates for that rule (default 1 for SQUAREM, 10 for Anderson, whose
                         single steps can be tiny on flat tails while the climb continues).
        """
        if init_theta is None:
            theta = self._activef / self._activef.sum(axis=1, keepdims=True)
        else:
            theta = self._compact(init_theta) * self._activef
            theta = theta / theta.sum(axis=1, keepdims=True)
        a = None
        if prior is not None:
            a = np.broadcast_to(np.asarray(prior, dtype=np.float64), (self.K, self.T_full))
            a = self._compact(a) * self._activef
        history = []
        converged = False
        n_steps = 0
        it = 0
        if accelerate == "auto":
            pooled = self._pooling is not None and self._pooling["level"] in ("transcript", "design", "design_gene")
            accelerate = "anderson" if pooled else "squarem"
        if accelerate == "anderson":
            theta, history, converged, n_steps = self._anderson(theta, a, max_iter, min_iter, rel_tol, theta_floor,
                                                                ll_rel_tol, verbose, ll_window=ll_window or 10)
        while accelerate != "anderson" and n_steps < max_iter:
            t1, ll0 = self._update(theta, a)
            t2, ll1 = self._update(t1, a)
            n_steps += 2
            new = t2
            if accelerate:
                r = t1 - theta
                v = t2 - t1
                v -= r
                nv = np.sqrt(np.einsum("kt,kt->", v, v))
                if nv > 0:
                    step = min(-np.sqrt(np.einsum("kt,kt->", r, r)) / nv, -1.0)
                    tp = r
                    tp *= -2.0 * step
                    tp += theta
                    v *= step * step
                    tp += v
                    np.maximum(tp, 1e-16, out=tp)
                    tp *= self._activef
                    tp /= tp.sum(axis=1, keepdims=True)
                    t3, llp = self._update(tp, a)
                    n_steps += 1
                    if llp >= ll1:
                        new = t3
            # a small non-negative gain (a decrease, e.g. after a rejected extrapolation, is not convergence)
            w = ll_window or 1
            ll_stall = (ll_rel_tol is not None and len(history) >= w
                        and 0.0 <= ll0 - history[-w] < w * ll_rel_tol * abs(ll0))
            history.append(ll0)
            diff = np.abs(new - theta)
            changed = int(np.count_nonzero((new > theta_floor) & (diff > rel_tol * new)))
            theta = new
            it += 1
            if verbose and (it <= 3 or it % 50 == 0):
                print(f"[JointEM] iterate {it} ({n_steps} MM steps): logpost={ll0:.6e} changed={changed}")
            if n_steps >= min_iter and (changed == 0 or ll_stall):
                converged = True
                break
        if zero_below is not None and self._pooling is not None and self._pooling["level"] in ("design", "design_gene"):
            zero_below = None              # the design prior gives unseen transcripts shrunk abundances
        if zero_below is not None:
            Z, _, _, _ = self._estep(theta)
            if self._tie is not None:              # zero by the tie group's expected reads
                tie = self._tie[0]
                Zj = np.stack([Z[tie == j].sum(axis=0) for j in range(int(tie.max()) + 1)])
                Z = Zj[tie]
            theta = np.where(Z >= zero_below, theta, 0.0)
            theta /= theta.sum(axis=1, keepdims=True)
            theta, _ = self._update(theta, None if a is None else np.where(theta > 0, a, 0.0))
            n_steps += 1
        _, _, ll, R = self._estep(theta, expected_reads=True)
        history.append(ll)
        return JointResult(theta=self._expand(theta), n_iter=n_steps, converged=converged, loglik=history,
                           expected_reads=self._expand(R))


def library_from_tcc(name, tcc, platform, eff_lens=None) -> Library:
    """Library with kallisto-style sampling rates: eff_lens for short reads, ones for long reads."""
    T = len(tcc.transcript_names)
    if platform == "short":
        if eff_lens is None:
            raise ValueError("short-read library needs eff_lens")
        b = np.asarray(eff_lens, dtype=np.float64)
    else:
        b = np.ones(T) if eff_lens is None else np.asarray(eff_lens, dtype=np.float64)
    return Library(name=name, tcc=tcc, b=b, platform=platform)


def molar_to_tpm(theta: np.ndarray) -> np.ndarray:
    """Molar fractions (..., T) to TPM."""
    return 1e6 * theta / theta.sum(axis=-1, keepdims=True)


def _newton_lognormal(x, Z, Dw, m, tau2, act, n_iter=4):
    """
    Elementwise maximiser of f(x) = Z x - Dw e^x - (x - m)^2 / (2 tau2) (concave), in closed form:
    the stationary point is x = c - W(tau2 Dw e^c) with c = m + tau2 Z and W the Lambert W function.
    w = W(e^y), y = log(tau2 Dw) + c, solves w + log w = y; Newton on that monotone, concave
    equation from w0 = log(1 + e^y) reaches 1e-15 relative error in 4 steps. Entries outside act keep x.
    """
    from . import native
    if native.available():
        xr = native.lognormal_solve(np.ravel(Z), np.ravel(Dw), np.ravel(np.broadcast_to(m, np.shape(Z))),
                                    tau2).reshape(np.shape(Z))
        return xr if (np.ndim(act) == 0 and act) else np.where(act, xr, x)
    c = m + tau2 * Z
    with np.errstate(divide="ignore"):
        y = np.log(tau2 * Dw) + c                          # -inf where Dw = 0
    small = y < -30.0                                       # W(z) = z - z^2 + ...: w ~ e^y
    yy = np.where(small, 0.0, y)
    w = np.where(yy > 30.0, yy, np.log1p(np.exp(np.minimum(yy, 30.0))))
    for _ in range(n_iter):
        w -= (w + np.log(w) - yy) / (1.0 + 1.0 / w)
        np.maximum(w, 1e-300, out=w)
    w = np.where(small, np.exp(np.clip(y, -745.0, 0.0)), w)
    return np.where(act, c - w, x)


def _design_newton(x, Z, D, c, Q, n_newton=8, tol=1e-9, chunk=20000):
    """
    Per-transcript maximiser of g_t(x) = Z_t . x - D_t . e^x - (x - c)' Q (x - c) / 2 over the K units
    (columns of x (K, T)), by damped Newton steps with batched K x K solves of diag(D e^x) + Q;
    concave, so each step is halved until g increases. Returns x (K, T).
    """
    x = x.copy()
    K, T = x.shape
    for t0 in range(0, T, chunk):
        sl = slice(t0, min(T, t0 + chunk))
        xs = x[:, sl].T.copy()                       # (n, K)
        Zs, Ds = Z[:, sl].T, D[:, sl].T
        def g(v):
            r = v - c
            return (Zs * v).sum(1) - (Ds * np.exp(np.minimum(v, 700.0))).sum(1) - 0.5 * np.einsum("nk,kj,nj->n", r, Q, r)
        gv = g(xs)
        for _ in range(n_newton):
            w = Ds * np.exp(np.minimum(xs, 700.0))
            grad = Zs - w - (xs - c) @ Q
            H = Q[None, :, :] + w[:, :, None] * np.eye(K)[None, :, :]
            H[:, np.arange(K), np.arange(K)] += 1e-12
            step = np.linalg.solve(H, grad[:, :, None])[:, :, 0]
            t = np.ones(len(xs))
            for _ in range(12):
                xn = xs + t[:, None] * step
                gn = g(xn)
                bad = gn < gv - 1e-12 * np.abs(gv)
                if not bad.any():
                    break
                t = np.where(bad, 0.5 * t, t)
            ok = gn >= gv - 1e-12 * np.abs(gv)
            xs = np.where(ok[:, None], xn, xs)
            gv = np.where(ok, gn, gv)
            if np.abs(step * t[:, None]).max() < tol:
                break
        x[:, sl] = xs.T
    return x
