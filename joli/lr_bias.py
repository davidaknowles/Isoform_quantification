"""
lr_bias.py
==========
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


def fit_with_truncation(em, models, n_outer=10, run_kw=None, verbose=False):
    """
    Alternate JointEM runs (warm-started) with truncation-model M-steps.

    Args:
        em     : JointEM whose long-read libraries carry Library.feat from TruncationModel.features.
        models : dict {library index: TruncationModel}.
        n_outer: number of outer iterations.
    Returns:
        JointResult of the final run; models are updated in place.
    """
    run_kw = dict(run_kw or {})
    for l, m in models.items():
        fl = em.libs[l]
        em.set_bias(l, c=m.weights(fl.tx, fl.feat))
    res = em.run(**run_kw)
    for it in range(n_outer):
        for l, m in models.items():
            fl = em.libs[l]
            m.mstep(fl.tx, fl.feat, em.position_counts(res.theta, l))
            em.set_bias(l, c=m.weights(fl.tx, fl.feat))
        # floor so transcripts zeroed by the previous run's final round can return
        res = em.run(init_theta=np.maximum(res.theta, 1e-10), **run_kw)
        if verbose:
            print(f"[truncation] outer {it + 1}: loglik={res.loglik[-1]:.6e} n_iter={res.n_iter}", flush=True)
    return res
