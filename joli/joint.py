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

from .load_tcc import TCCData

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

    def __init__(self, lib: Library):
        counts = np.asarray(lib.tcc.ec_counts, dtype=np.float64)
        keep = np.flatnonzero(counts > 0)
        sizes = np.array([len(lib.tcc.ec_transcripts[e]) for e in keep], dtype=np.int64)
        self.tx = np.concatenate([np.asarray(lib.tcc.ec_transcripts[e], dtype=np.int64) for e in keep])
        self.ec = np.repeat(np.arange(len(keep), dtype=np.int64), sizes)
        if lib.c is None:
            self.c = np.ones(len(self.tx), dtype=np.float64)
        else:
            self.c = np.concatenate([np.asarray(lib.c[e], dtype=np.float64) for e in keep])
        self.feat = None if lib.feat is None else \
            np.concatenate([np.asarray(lib.feat[e]).reshape(sizes[i], -1) for i, e in enumerate(keep)])
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

    Args:
        libraries : list[Library]
        pi        : np.ndarray (L, K) | None -- library x unit molecule fractions;
                    None means each library is its own unit (K = L).
        units     : list[int] | None -- shorthand for one-hot pi: units[l] = k.
        unit_names: list[str] | None
        backend   : "auto" | "numpy" | "rust" -- E-step implementation; "auto" uses the
                    compiled kernels (joli.native, rust/joli_kernels) when built.
        threads   : int | None -- threads for the compiled kernels.
    """

    def __init__(self, libraries, pi=None, units=None, unit_names=None, backend="auto", threads=None):
        from . import native
        if backend == "auto":
            backend = "rust" if native.available() else "numpy"
        if backend == "rust" and not native.available():
            raise RuntimeError("rust backend requested but rust/joli_kernels is not built")
        self.backend, self.threads = backend, threads
        self.libs = [_FlatLibrary(l) for l in libraries]
        self.L = len(self.libs)
        self.T = len(libraries[0].tcc.transcript_names)
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
        # transcripts that appear in a nonzero EC of some library of unit k
        self.active = np.zeros((self.K, self.T), dtype=bool)
        for l, fl in enumerate(self.libs):
            seen = np.zeros(self.T, dtype=bool)
            seen[fl.tx] = True
            self.active[self.pi[l] > 0] |= seen

    def _library_estep(self, l, mu, threads):
        """(g, sum_e n_e log denom_e) for library l at molar mixture mu."""
        fl = self.libs[l]
        if self.backend == "rust":
            from . import native
            if threads > 1:
                if getattr(fl, "gather", None) is None:
                    fl.gather = native.gather_index(fl.tx, fl.ec, self.T)
                if getattr(fl, "c_sorted", None) is None:      # reset by set_bias
                    fl.c_sorted = np.ascontiguousarray(fl.c[fl.gather[1]])
                return native.estep(fl.ec_ptr, fl.tx, fl.c, fl.counts, mu, threads, fl.gather, fl.c_sorted)
            return native.estep(fl.ec_ptr, fl.tx, fl.c, fl.counts, mu, 1)
        w = mu[fl.tx] * fl.c                                    # per position
        denom = np.bincount(fl.ec, weights=w, minlength=fl.n_ec)
        ok = denom > TOLERANCE
        ratio = np.where(ok, fl.counts / np.where(ok, denom, 1.0), 0.0)
        r_pos = fl.c * ratio[fl.ec]                             # n_e c_et / denom_e
        g = np.bincount(fl.tx, weights=r_pos, minlength=self.T)  # sum_e n_e c_et / denom_e
        return g, float(fl.counts[ok] @ np.log(denom[ok]))

    def _estep(self, theta, expected_reads=False):
        """
        Return Z (K, T), D (K, T), loglik, and expected reads per library (L, T)
        if requested (else None). With the rust backend and several threads,
        libraries run concurrently (ctypes releases the GIL); a single library
        is split across threads only if it has >= 5M positions per thread.
        """
        Z = np.zeros((self.K, self.T))
        D = np.zeros((self.K, self.T))
        R = np.zeros((self.L, self.T)) if expected_reads else None
        mus = [self.pi[l, self.pi[l] > 0] @ theta[self.pi[l] > 0] for l in range(self.L)]
        threads = 1
        if self.backend == "rust":
            from . import native
            threads = self.threads or native.default_threads()
        if threads > 1 and self.L > 1 and sum(len(fl.tx) for fl in self.libs) >= 1_000_000:
            if getattr(self, "_pool", None) is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(max_workers=threads)
            out = list(self._pool.map(lambda l: self._library_estep(l, mus[l], 1), range(self.L)))
        else:
            out = [self._library_estep(l, mus[l], min(threads, max(1, len(fl.tx) // 5_000_000)))
                   for l, fl in enumerate(self.libs)]
        loglik = 0.0
        for l, fl in enumerate(self.libs):
            g, ll = out[l]
            mu = mus[l]
            B = float(mu @ fl.b)
            loglik += ll - fl.N * np.log(B)
            if expected_reads:
                R[l] = mu * g
            for k in np.flatnonzero(self.pi[l] > 0):
                Z[k] += self.pi[l, k] * theta[k] * g
                D[k] += fl.N * self.pi[l, k] * fl.b / B
        return Z, D, loglik, R

    def position_counts(self, theta, l):
        """
        Expected reads per (EC, transcript) position of library l,
        z_lp = n_e mu_t c_et / sum_{t' in e} mu_t' c_et', as a flat array
        aligned with the library's flat EC layout (for M-steps of bias models).
        """
        fl = self.libs[l]
        ks = np.flatnonzero(self.pi[l] > 0)
        mu = self.pi[l, ks] @ theta[ks]
        if self.backend == "rust":
            from . import native
            return native.position_counts(fl.ec_ptr, fl.tx, fl.c, fl.counts, mu,
                                          min(self.threads or native.default_threads(), max(1, len(fl.tx) // 1_000_000)))
        w = mu[fl.tx] * fl.c
        denom = np.bincount(fl.ec, weights=w, minlength=fl.n_ec)
        return np.where(denom[fl.ec] > TOLERANCE, fl.counts[fl.ec] * w / np.maximum(denom[fl.ec], TOLERANCE), 0.0)

    def set_bias(self, l, c=None, b=None):
        """Replace library l's flat compatibility weights c (aligned with its flat layout) and/or rates b."""
        if c is not None:
            self.libs[l].c = np.ascontiguousarray(c, dtype=np.float64)
            self.libs[l].c_sorted = None
        if b is not None:
            self.libs[l].b = np.ascontiguousarray(b, dtype=np.float64)

    def _update(self, theta, a):
        """One MM step. Returns (new theta, log posterior at the input theta)."""
        Z, D, ll, _ = self._estep(theta)
        if a is None:
            num = Z
        else:
            num = Z + a
            ll += float((a[theta > 0] * np.log(theta[theta > 0])).sum())
        num = np.where(self.active, num, 0.0)
        # lambda_k from sum_t theta_kt = 1 at the fixed point: sum(num) - sum(theta_old * D)
        lam = num.sum(axis=1, keepdims=True) - (theta * D).sum(axis=1, keepdims=True)
        new = np.where(self.active, num / np.maximum(D + lam, TOLERANCE), 0.0)
        return new / new.sum(axis=1, keepdims=True), ll

    def run(self, prior=None, init_theta=None, max_iter=5000, min_iter=20,
            rel_tol=1e-2, theta_floor=1e-8, accelerate=True, zero_below=0.01,
            verbose=False) -> JointResult:
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
            prior      : np.ndarray (T,) or (K, T) | None -- Dirichlet pseudo-counts a.
            init_theta : np.ndarray (K, T) | None -- warm start; uniform over active otherwise.
            zero_below : float | None -- expected-read threshold for final zeroing.
        """
        if init_theta is None:
            theta = self.active / self.active.sum(axis=1, keepdims=True)
        else:
            theta = np.where(self.active, init_theta, 0.0)
            theta = theta / theta.sum(axis=1, keepdims=True)
        a = None if prior is None else np.broadcast_to(np.asarray(prior, dtype=np.float64), (self.K, self.T))
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
                v = t2 - t1 - r
                nv = np.sqrt((v * v).sum())
                if nv > 0:
                    step = min(-np.sqrt((r * r).sum()) / nv, -1.0)
                    tp = theta - 2.0 * step * r + step * step * v
                    tp = np.where(self.active, np.maximum(tp, 1e-16), 0.0)
                    tp /= tp.sum(axis=1, keepdims=True)
                    t3, llp = self._update(tp, a)
                    n_steps += 1
                    if llp >= ll1:
                        new = t3
            history.append(ll0)
            mon = new > theta_floor
            changed = int((mon & (np.abs(new - theta) > rel_tol * new)).sum())
            theta = new
            it += 1
            if verbose and (it <= 3 or it % 50 == 0):
                print(f"[JointEM] iterate {it} ({n_steps} MM steps): logpost={ll0:.6e} changed={changed}")
            if n_steps >= min_iter and changed == 0:
                converged = True
                break
        if zero_below is not None:
            Z, _, _, _ = self._estep(theta)
            theta = np.where(Z >= zero_below, theta, 0.0)
            theta /= theta.sum(axis=1, keepdims=True)
            theta, _ = self._update(theta, None if a is None else np.where(theta > 0, a, 0.0))
            n_steps += 1
        _, _, ll, R = self._estep(theta, expected_reads=True)
        history.append(ll)
        return JointResult(theta=theta, n_iter=n_steps, converged=converged, loglik=history, expected_reads=R)


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
