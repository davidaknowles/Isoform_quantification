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
        pool      : None | "gene" | "transcript" -- partial pooling across units with the same tie
                    label instead of exact sharing. "gene": theta_ut = G_ug psi_jt with shared psi and
                    log G_ug ~ N(log H_jg + s_u, pool_tau^2); "transcript": log theta_ut ~ N(eta_jt + c_u,
                    pool_tau^2). Small pool_tau approaches a shared theta, large pool_tau separate units
                    (gene: shared psi with free gene totals). MAP by MM with Newton steps on the logs.
        backend   : "auto" | "numpy" | "rust" -- E-step implementation; "auto" uses the
                    compiled kernels (joli.native, rust/joli_kernels) when built.
        threads   : int | None -- threads for the compiled kernels.
    """

    def __init__(self, libraries, pi=None, units=None, unit_names=None, backend="auto", threads=None,
                 sort_ecs=True, tie=None, groups=None, tie_iter=1, pool=None, pool_tau=1.0):
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
        self._pool = None
        if pool is not None:
            if tie is None or (pool == "gene" and groups is None):
                raise ValueError("pool needs tie (and groups for gene-level pooling)")
            self._pool = {"level": pool, "tau2": float(pool_tau) ** 2, "state": None}
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
            self._tie = (pd.factorize(tie)[0], gc.astype(np.int64), int(gc.max()) + 1, int(tie_iter))
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
        if self._tie is not None:
            if a is not None:
                raise NotImplementedError("tied fractions with a Dirichlet prior")
            if self._pool is not None and self._pool["level"] == "transcript":
                new, lp = self._pooled_transcript_mstep(theta, Z, D)
                return new, ll + lp
            if self._pool is not None:
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
            st = self._pool["state"]
            if st is None:
                st = self._lognormal_profile(y, gact, tie, M)
            lp = self._lognormal_logprior(y, st, gact, tie)
        for _ in range(n_iter):
            if pooled:
                Dg = agg(D * psi[tie])
                y = _newton_lognormal(y, Zg, Dg, st[0][tie] + st[1][:, None], self._pool["tau2"], gact)
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
            self._pool["state"] = st
            return new, lp
        return new

    def _lognormal_profile(self, y, act, tie, M):
        """Shared profile h_j = mean_{u in j}(y_u - s_u) and unit offsets s_u = mean_t (y_ut - h_jt) over
        active entries (a few alternating rounds; mean of s within each tie group fixed at 0)."""
        a = act.astype(np.float64)
        s = np.zeros(self.K)
        for _ in range(3):
            h = np.divide(M @ (np.where(act, y, 0.0) - s[:, None] * a), M @ a,
                          out=np.zeros((M.shape[0], y.shape[1])), where=(M @ a) > 0)
            r = np.where(act, y - h[tie], 0.0)
            s = r.sum(axis=1) / np.maximum(a.sum(axis=1), 1.0)
            sm = (M @ s) / np.maximum(M.sum(axis=1), 1.0)
            s -= sm[tie]
        return h, s

    def _lognormal_logprior(self, y, st, act, tie):
        r = np.where(act, y - st[0][tie] - st[1][:, None], 0.0)
        return float(-0.5 * (r * r).sum() / self._pool["tau2"])

    def _pooled_transcript_mstep(self, theta, Z, D):
        """
        MAP-MM step for log theta_ut ~ N(eta_jt + c_u, tau^2) (j = tie(u)): Newton steps on
        x = log theta for Q = sum Z x - D e^x + log prior, then the profile (eta, c). Units of a tie
        group share their active set, so transcripts without reads in a unit get abundance from
        the other units of its group through eta.
        """
        tie = self._tie[0]
        if getattr(self, "_tie_mats", None) is None or self._tie_mats[1].shape[1] != self.K:
            M = np.zeros((int(tie.max()) + 1, self.K))
            M[tie, np.arange(self.K)] = 1.0
            self._tie_mats = (None, M)
        M = self._tie_mats[1]
        act = self.active
        x = np.log(np.maximum(theta, 1e-300))
        st = self._pool["state"] or self._lognormal_profile(x, act, tie, M)
        lp = self._lognormal_logprior(x, st, act, tie)
        x = _newton_lognormal(x, Z, D, st[0][tie] + st[1][:, None], self._pool["tau2"], act)
        st = self._lognormal_profile(x, act, tie, M)
        self._pool["state"] = st
        x = np.where(act, x, -np.inf)
        new = np.exp(x - x.max(axis=1, keepdims=True))
        new /= new.sum(axis=1, keepdims=True)
        return new, lp

    def run(self, prior=None, init_theta=None, max_iter=5000, min_iter=20,
            rel_tol=1e-2, theta_floor=1e-8, accelerate=True, zero_below=0.01,
            ll_rel_tol=None, verbose=False) -> JointResult:
        """
        Iterate until no monitored theta (> theta_floor) changes by more than
        rel_tol relatively between successive iterates (kallisto's rule; its
        0.01-read floor corresponds to theta_floor).

        With accelerate=True each iterate is a SQUAREM step (Varadhan and
        Roland 2008, scheme S3): two MM steps, an extrapolation, and one
        stabilising MM step, falling back to the plain second MM step when the
        extrapolated point has lower log posterior. n_iter counts MM steps.

        After convergence, transcripts of unit k whose expected read count over
        all libraries is below zero_below are set to zero and one more MM step
        is run (kallisto's finalRound). MLEs on the simplex boundary otherwise
        stay at small positive values whose number depends on the iteration count.

        Args:
            prior      : np.ndarray (T,) or (K, T) | None -- Dirichlet pseudo-counts a (full length).
            init_theta : np.ndarray (K, T) | None -- warm start (full length); uniform over active otherwise.
            zero_below : float | None -- expected-read threshold for final zeroing.
            ll_rel_tol : float | None -- also stop when an iterate raises the log posterior by less
                         than ll_rel_tol * |log posterior| (weakly identified fits, where theta drifts
                         along nearly flat directions long after the likelihood has settled).
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
        while n_steps < max_iter:
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
            ll_stall = (ll_rel_tol is not None and len(history) > 0
                        and ll0 - history[-1] < ll_rel_tol * abs(ll0))
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


def _newton_lognormal(x, Z, Dw, m, tau2, act, n_newton=4):
    """
    Elementwise maximisation of f(x) = Z x - Dw e^x - (x - m)^2 / (2 tau2) (concave) by Newton
    steps with step halving, from x; entries outside act are left unchanged.
    """
    f = lambda v: Z * v - Dw * np.exp(np.minimum(v, 700.0)) - (v - m) ** 2 / (2.0 * tau2)
    x = np.where(act, np.maximum(x, m - 50.0), x)
    fx = f(x)
    for _ in range(n_newton):
        ex = np.exp(np.minimum(x, 700.0))
        g = Z - Dw * ex - (x - m) / tau2
        h = -Dw * ex - 1.0 / tau2
        step = -g / h
        t = np.ones_like(x)
        for _ in range(8):
            xn = x + t * step
            fn = f(xn)
            bad = act & (fn < fx)
            if not bad.any():
                break
            t = np.where(bad, 0.5 * t, t)
        ok = act & (fn >= fx)
        x = np.where(ok, xn, x)
        fx = np.where(ok, fn, fx)
    return x
