"""
lr_bias.py
==========
Bias models: read-end truncation for long reads, read position for
single-cell short reads, and per-library length-dependent sampling.

Read-end (truncation) model for long reads.

A long read aligned to transcript t (length L_t) at [s, e) leaves d5 = s
bases uncovered at the 5' end and d3 = L_t - e at the 3' end. Poly(A)-primed
cDNA reads are 3'-anchored and 5'-truncated to a variable degree; cap-trapped
or direct RNA libraries have other profiles. We model

    P(d5, d3 | t) = q5(d5 | L_t) q3(d3 | L_t),   q(d | L) = (p_j / w_j) / F(L)  for d in bin j,

with p (J,) a distribution over distance bins of width w_j, uniform within
a bin, and F(L) = sum_j p_j o_j(L) the mass of distances that fit in a
transcript of length L (o_j(L) the fraction of bin j below L + 1). The
weights enter JointEM as c_et = q5 q3, so a read covering the 3' kilobase of
a long and a short isoform favours the short one when long truncations are
rare.

Given expected reads z per (read, transcript) alignment, the M-step for p
maximises sum_j A_j log p_j - sum_t Z_t log F_t(p), with A_j the expected
reads in bin j and Z_t the expected reads of transcript t. Its stationarity
condition gives the fixed point p_j = A_j / sum_t Z_t o_tj / F_t (the EM for
truncated multinomial data), iterated to convergence.
"""

import numpy as np

DEFAULT_EDGES = np.array([0, 1, 3, 10, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000,
                          1500, 2000, 3000, 5000, 10000, 200000], dtype=np.float64)


class EndProfile:
    """Binned distribution of uncovered bases at one transcript end."""

    def __init__(self, edges=DEFAULT_EDGES):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.width = np.diff(self.edges)
        self.J = len(self.width)
        self.p = np.full(self.J, 1.0 / self.J)

    def bin(self, d):
        """Bin index of distances d (n,)."""
        return np.clip(np.searchsorted(self.edges, d, side="right") - 1, 0, self.J - 1)

    def overlap(self, tx_len):
        """o (T, J): fraction of each bin's integer distances 0..L_t inside the transcript."""
        L = np.asarray(tx_len, dtype=np.float64)[:, None] + 1.0
        return np.clip((np.minimum(L, self.edges[1:]) - self.edges[:-1]) / self.width, 0.0, 1.0)

    def weight(self, bins, tx, F):
        """q(d | L_t) for alignments with bin index `bins` on transcripts `tx`; F (T,) from norm()."""
        return self.p[bins] / self.width[bins] / F[tx]

    def norm(self, O):
        return np.maximum(O @ self.p, 1e-300)

    def mstep(self, A, Z, O, n_iter=200, tol=1e-10, pseudo=1e-3):
        """
        Fixed-point M-step.

        Args:
            A : (J,) expected reads per bin.
            Z : (T,) expected reads per transcript.
            O : (T, J) overlap matrix.
            pseudo : float -- pseudo-count per bin keeping p_j > 0.
        """
        A = A + pseudo
        act = Z > 0
        Za, Oa = Z[act], O[act]
        p = self.p.copy()
        support = (Za @ Oa) > 0               # bins reachable by some expressed transcript
        A = np.where(support, A, 0.0)
        p = np.where(support, p, 0.0)
        p /= p.sum()
        for _ in range(n_iter):
            F = np.maximum(Oa @ p, 1e-300)
            denom = (Za / F) @ Oa
            new = np.where(support, A / np.where(support, denom, 1.0), 0.0)
            new /= new.sum()
            done = np.abs(new - p).max() < tol
            p = new
            if done:
                break
        self.p = p


class TruncationModel:
    """5' and 3' end profiles for one long-read library."""

    def __init__(self, tx_len, edges5=DEFAULT_EDGES, edges3=DEFAULT_EDGES):
        self.tx_len = np.asarray(tx_len, dtype=np.float64)
        self.end5 = EndProfile(edges5)
        self.end3 = EndProfile(edges3)
        self.O5 = self.end5.overlap(self.tx_len)
        self.O3 = self.end3.overlap(self.tx_len)

    def features(self, tstart, tend, tlen):
        """Integer features (n, 2): bin of d5 = tstart and of d3 = tlen - tend."""
        tstart, tend, tlen = (np.asarray(x, dtype=np.float64) for x in (tstart, tend, tlen))
        return np.column_stack([self.end5.bin(tstart), self.end3.bin(tlen - tend)])

    def weights(self, tx, feat):
        """c (n,) for alignments to transcripts tx with features feat (n, 2)."""
        F5 = self.end5.norm(self.O5)
        F3 = self.end3.norm(self.O3)
        return self.end5.weight(feat[:, 0], tx, F5) * self.end3.weight(feat[:, 1], tx, F3)

    def mstep(self, tx, feat, z):
        """Update both profiles from expected reads z (n,) of alignments (tx, feat)."""
        T = len(self.tx_len)
        Z = np.bincount(tx, weights=z, minlength=T)
        self.end5.mstep(np.bincount(feat[:, 0], weights=z, minlength=self.end5.J), Z, self.O5)
        self.end3.mstep(np.bincount(feat[:, 1], weights=z, minlength=self.end3.J), Z, self.O3)


REL_EDGES = np.linspace(0.0, 1.0, 21)
D3_EDGES = np.array([0, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000,
                     5000, 10000, 30000, 100000, 3e6], dtype=np.float64)


class ReadPositionModel:
    """
    Distribution of short-read positions along transcripts, for single-cell
    libraries where read position depends on the RT primer.

    A read of length r on transcript t (length L_t) starts at s in
    {0, ..., L_t - r}. Two parametrisations, each a distribution p over bins:
      mode "3p"       : bins of d3 = L_t - (s + r), the bases between the read and
                        the 3' end (poly(dT) priming: fragments hang off the poly(A)
                        tail), normalised over the d3 values that fit, as EndProfile.
      mode "relative" : bins of u = s / (L_t - r) on [0, 1] (random-hexamer priming),
                        q(s | t) = p_j / (w_j (L_t - r + 1)).
    weights() returns the density q(s | t); fit_bias multiplies it by the library's
    sampling rate b_t, so a uniform profile reproduces the position-free model when
    b_t = L_t - r + 1.
    """

    def __init__(self, tx_len, read_len=150, mode="3p", edges=None):
        self.mode = mode
        self.span = np.maximum(np.asarray(tx_len, dtype=np.float64) - read_len, 0.0)  # max start / max d3
        if mode == "3p":
            self.prof = EndProfile(D3_EDGES if edges is None else edges)
            self.O = self.prof.overlap(self.span)
        elif mode == "relative":
            self.prof = EndProfile(REL_EDGES if edges is None else edges)
            self.O = None
        else:
            raise ValueError(mode)

    def features(self, start, end, tlen):
        """Integer feature (n, 1): bin of d3 or of the relative start."""
        start, end, tlen = (np.asarray(x, dtype=np.float64) for x in (start, end, tlen))
        if self.mode == "3p":
            return self.prof.bin(np.maximum(tlen - end, 0.0))[:, None]
        rel = start / np.maximum(tlen - (end - start), 1.0)
        return np.clip(np.searchsorted(self.prof.edges, rel, side="right") - 1, 0, self.prof.J - 1)[:, None]

    def weights(self, tx, feat):
        j = feat[:, 0]
        if self.mode == "3p":
            return self.prof.weight(j, tx, self.prof.norm(self.O))
        return self.prof.p[j] / self.prof.width[j] / (self.span[tx] + 1.0)

    def mstep(self, tx, feat, z):
        A = np.bincount(feat[:, 0], weights=z, minlength=self.prof.J)
        if self.mode == "3p":
            self.prof.mstep(A, np.bincount(tx, weights=z, minlength=len(self.span)), self.O)
        else:
            p = A + 1e-3
            self.prof.p = p / p.sum()


class LengthBias:
    """
    Library-specific sampling rate as a smooth function of transcript length,
    b_t = exp(x_t . w), with x_t (P,) a piecewise-linear basis in log length
    (log L and hinges at the knots). It multiplies the library's base rates
    (1 for long reads, effective length for short reads). A read of transcript
    t then has P(r, t) proportional to mu_t b_t q(r | t), so b also multiplies c.

    b is confounded with theta unless theta is shared with another library
    whose rates are fixed (e.g. SR with effective lengths, or a reference LR
    protocol). The M-step maximises sum_t Z_t log b_t - N log sum_t mu_t b_t
    (concave in w) by Newton steps with a ridge penalty.
    """

    def __init__(self, tx_len, knots=(500, 1000, 2000, 4000, 8000), ridge=1.0):
        self.knots = tuple(knots)
        X = self._basis(tx_len)
        self.center = X.mean(axis=0)
        self.X = X - self.center
        self.w = np.zeros(self.X.shape[1])
        self.ridge = ridge

    def _basis(self, lengths):
        logL = np.log(np.maximum(np.asarray(lengths, dtype=np.float64), 1.0))
        return np.column_stack([logL] + [np.maximum(0.0, logL - np.log(k)) for k in self.knots])

    def rates(self):
        return np.exp(self.X @ self.w)

    def rates_at(self, lengths):
        """b at arbitrary lengths, on the same scale as rates()."""
        return np.exp((self._basis(lengths) - self.center) @ self.w)

    def mstep(self, Z, mu, n_newton=20):
        """
        Z (T,) expected reads per transcript and mu (T,) molar mixture of one
        library, or lists of them for libraries sharing this length bias: the
        objective is then sum_l [Z_l . log b - N_l log sum_t mu_lt b_t].
        """
        Zs = Z if isinstance(Z, (list, tuple)) else [Z]
        mus = mu if isinstance(mu, (list, tuple)) else [mu]
        Ns = [z.sum() for z in Zs]
        XZ = sum(self.X.T @ z for z in Zs)
        P = len(self.w)
        for _ in range(n_newton):
            r = self.rates()
            grad = XZ - self.ridge * self.w
            H = self.ridge * np.eye(P)
            for N, m_ in zip(Ns, mus):
                pb = m_ * r
                p = pb / pb.sum()
                m = self.X.T @ p
                grad -= N * m
                H += N * ((self.X * p[:, None]).T @ self.X - np.outer(m, m))
            step = np.linalg.solve(H, grad)
            self.w += step
            if np.abs(step).max() < 1e-8:
                break


class FeatureColumns:
    """
    Adapter so a positional model reads selected columns of a library's
    stacked features (several models' features stored side by side, so ECs
    are exact for each).
    """

    def __init__(self, model, cols):
        self.model, self.cols = model, list(cols)

    def features(self, start, end, tlen):
        return self.model.features(start, end, tlen)

    def weights(self, tx, feat):
        return self.model.weights(tx, feat[:, self.cols])

    def mstep(self, tx, feat, z):
        self.model.mstep(tx, feat[:, self.cols], z)


class StackedFeatures:
    """Concatenated integer features of several positional models."""

    def __init__(self, models):
        self.models = list(models)

    def features(self, start, end, tlen):
        return np.column_stack([m.features(start, end, tlen) for m in self.models])


def fit_bias(em, trunc=None, length=None, n_outer=10, run_kw=None, verbose=False):
    """
    Alternate JointEM runs (warm-started) with bias-model M-steps.

    Args:
        em     : JointEM. Libraries in `trunc` carry Library.feat from TruncationModel.features.
                 Each library's initial c and b (e.g. alignment-score weights, effective
                 lengths) are kept as base values that the bias factors multiply.
        trunc  : dict {library index: positional model} -- TruncationModel (long reads) or
                 ReadPositionModel (short reads); any object with features/weights/mstep.
                 The library's c becomes base c x b_t x q(position | t). Libraries mapped to
                 the same model object share it (one pooled M-step).
        length : dict {library index: LengthBias}, shared in the same way.
        n_outer: number of outer iterations.
    Returns:
        JointResult of the final run; models are updated in place.
    """
    trunc, length = trunc or {}, length or {}
    run_kw = dict(run_kw or {})
    libs = set(trunc) | set(length)
    base_c = {l: em.libs[l].c.copy() for l in libs}
    base_b = {l: em.libs[l].b.copy() for l in libs}

    def apply():
        for l in libs:
            fl = em.libs[l]
            c = base_c[l].copy()
            b = base_b[l].copy()
            if l in trunc:
                # positional density times the base sampling rate (c = b q)
                c *= trunc[l].weights(fl.tx, fl.feat) * base_b[l][fl.tx]
            if l in length:
                r = length[l].rates()
                c *= r[fl.tx]
                b *= r
            em.set_bias(l, c=c, b=b)

    apply()
    res = em.run(**run_kw)
    for it in range(n_outer):
        # one M-step per model object, pooling the libraries that share it
        for model, ls in _by_model(trunc).items():
            fls = [em.libs[l] for l in ls]
            model.mstep(np.concatenate([fl.tx for fl in fls]), np.concatenate([fl.feat for fl in fls]),
                        np.concatenate([em.position_counts(res.theta, l) for l in ls]))
        for model, ls in _by_model(length).items():
            mus = []
            for l in ls:
                ks = np.flatnonzero(em.pi[l] > 0)
                mus.append((em.pi[l, ks] @ res.theta[ks]) * base_b[l])
            model.mstep([res.expected_reads[l] for l in ls], mus)
        apply()
        # floor so transcripts zeroed by the previous run's final round can return
        res = em.run(init_theta=np.maximum(res.theta, 1e-10), **run_kw)
        if verbose:
            print(f"[bias] outer {it + 1}: loglik={res.loglik[-1]:.6e} n_iter={res.n_iter}", flush=True)
    return res


def _by_model(d):
    """Group library indices by model object identity."""
    groups = {}
    for l, m in d.items():
        groups.setdefault(id(m), (m, []))[1].append(l)
    return {m: ls for m, ls in groups.values()}


def fit_with_truncation(em, models, n_outer=10, run_kw=None, verbose=False):
    """fit_bias with truncation models only (kept for existing callers)."""
    return fit_bias(em, trunc=models, n_outer=n_outer, run_kw=run_kw, verbose=verbose)
