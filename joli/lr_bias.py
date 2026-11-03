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
import pandas as pd

DEFAULT_EDGES = np.array([0, 1, 3, 10, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000,
                          1500, 2000, 3000, 5000, 10000, 30000], dtype=np.float64)


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
        if not support.any():
            return
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


LEN_EDGES = np.array([0, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000,
                      4000, 5000, 7500, 10000, 30000], dtype=np.float64)


class TruncationModel:
    """
    5' and 3' end profiles for one long-read library.

    Targets in `uniform` (e.g. unspliced gene-span targets) get a read placed
    uniformly with a length drawn from a read-length distribution g:
    P(start, length | t) = g(length) / (L_t + 1), a density over the same two
    read-geometry coordinates as q5(d5) q3(d3) on other targets. g is learned
    from reads assigned to those targets (alignment length is a third feature
    when any target is uniform), and they are left out of the end-profile
    M-step: read-end distances on a pre-mRNA span say nothing about truncation,
    and profile mass at distances only those long targets reach is otherwise
    unidentified and can drift, shrinking their weights and starving them of reads.
    """

    def __init__(self, tx_len, edges5=DEFAULT_EDGES, edges3=DEFAULT_EDGES, uniform=None, len_edges=LEN_EDGES):
        self.tx_len = np.asarray(tx_len, dtype=np.float64)
        self.end5 = EndProfile(edges5)
        self.end3 = EndProfile(edges3)
        self.O5 = self.end5.overlap(self.tx_len)
        self.O3 = self.end3.overlap(self.tx_len)
        self.uniform = np.zeros(len(self.tx_len), dtype=bool) if uniform is None else np.asarray(uniform, dtype=bool)
        self.length = EndProfile(len_edges)              # read-length distribution on uniform targets

    def features(self, tstart, tend, tlen):
        """Integer features: bins of d5 = tstart and d3 = tlen - tend, plus alignment length if any target is uniform."""
        tstart, tend, tlen = (np.asarray(x, dtype=np.float64) for x in (tstart, tend, tlen))
        cols = [self.end5.bin(tstart), self.end3.bin(tlen - tend)]
        if self.uniform.any():
            cols.append(self.length.bin(tend - tstart))
        return np.column_stack(cols)

    def weights(self, tx, feat):
        """c (n,) for alignments to transcripts tx with features feat."""
        F5 = self.end5.norm(self.O5)
        F3 = self.end3.norm(self.O3)
        w = self.end5.weight(feat[:, 0], tx, F5) * self.end3.weight(feat[:, 1], tx, F3)
        u = self.uniform[tx]
        if u.any():
            lb = feat[u, 2]
            w[u] = self.length.p[lb] / self.length.width[lb] / (self.tx_len[tx[u]] + 1.0)
        return w

    def mstep(self, tx, feat, z):
        """Update the end profiles from alignments to non-uniform targets and g from uniform ones."""
        u = self.uniform[tx]
        if u.any():
            g = np.bincount(feat[u, 2], weights=z[u], minlength=self.length.J) + 1e-3
            self.length.p = g / g.sum()
        keep = ~u
        tx, feat, z = tx[keep], feat[keep], z[keep]
        T = len(self.tx_len)
        Z = np.bincount(tx, weights=z, minlength=T)
        self.end5.mstep(np.bincount(feat[:, 0], weights=z, minlength=self.end5.J), Z, self.O5)
        self.end3.mstep(np.bincount(feat[:, 1], weights=z, minlength=self.end3.J), Z, self.O3)


class AnchoredTruncationModel:
    """
    Read-end model for long reads: one end anchored, the other set by a
    censored molecule length.

    A 3'-anchored read (poly(dT)-primed cDNA, truncated towards the 5' end)
    ends d3 bases from the 3' end, d3 ~ q_a (an EndProfile normalised over
    d3 <= L_t), and has molecule length R ~ g over the m = L_t - d3 bases
    available. If R > m - delta the read reaches the 5' end (full length) and
    d5 ~ e, a distribution over [0, delta); otherwise the read has length
    r = m - d5 = R. So

        P(d5, d3 | t) = q_a(d3 | L_t) [ S(m - delta) e(d5) 1{d5 < delta} + g(r) 1{d5 >= delta} ],

    with g a binned density over read length (bins `len_edges`, uniform within
    a bin) and S its survival function. Full-length fractions fall with
    transcript length through S, which independent 5' and 3' profiles cannot
    express (they push mass to distances only long transcripts reach). A
    5'-anchored component (e.g. cap-trapped libraries) swaps the two ends. An
    "internal" component places reads of length r ~ g_i (truncated at L_t)
    uniformly: P = g_i(r) / G_i(L_t) / (L_t - r + 1) (fragments, both ends
    away from the transcript ends). The model is a mixture over `anchors` with
    weights pi, learned by default.

    Features per alignment: (d5 bin, d3 bin, aligned-length bin); bins of
    d5/d3 use `edges`, which must contain delta. Distances within a d5/d3 bin
    enter m at the bin midpoint. Targets in `uniform` (unspliced gene spans)
    get g_u(r) / (L_t + 1) with g_u a separate read-length density, and are
    left out of the other updates (see TruncationModel).

    M-step given expected reads z per alignment: component responsibilities;
    pi from their totals; q_a by the truncated-multinomial fixed point; e as a
    weighted histogram; g as the self-consistency (Turnbull) estimate of a
    right-censored binned density, with full-length reads censored at m - delta.
    """

    def __init__(self, tx_len, anchors=("3p", "5p"), pi=None, edges=DEFAULT_EDGES, len_edges=LEN_EDGES,
                 delta=50.0, uniform=None, learn_pi=True):
        self.tx_len = np.asarray(tx_len, dtype=np.float64)
        self.edges = np.asarray(edges, dtype=np.float64)
        if delta not in self.edges:
            raise ValueError("delta must be one of the d5/d3 bin edges")
        self.delta = float(delta)
        self.kf = int(np.flatnonzero(self.edges == delta)[0])       # d5/d3 bins below delta
        self.width = np.diff(self.edges)
        self.mid = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.anchors = tuple(anchors)
        self.learn_pi = learn_pi
        K = len(self.anchors)
        self.pi = np.full(K, 1.0 / K) if pi is None else np.asarray(pi, dtype=np.float64) / np.sum(pi)
        self.anchor = [EndProfile(self.edges) for _ in range(K)]
        for a in self.anchor:                                       # start concentrated near the anchor
            a.p = np.exp(-self.mid / 100.0)
            a.p /= a.p.sum()
        self.e = [np.full(self.kf, 1.0 / self.kf) for _ in range(K)]
        self.g = [EndProfile(len_edges) for _ in range(K)]
        self.O = self.anchor[0].overlap(self.tx_len)
        self.uniform = np.zeros(len(self.tx_len), dtype=bool) if uniform is None else np.asarray(uniform, dtype=bool)
        self.length = EndProfile(len_edges)                         # read length on uniform targets
        self.Olen = self.length.overlap(self.tx_len)                # read-length bins that fit in L_t

    def features(self, tstart, tend, tlen):
        """Integer features (n, 3): bins of d5 = tstart, d3 = tlen - tend, and aligned length."""
        tstart, tend, tlen = (np.asarray(x, dtype=np.float64) for x in (tstart, tend, tlen))
        b = self.anchor[0]
        return np.column_stack([b.bin(tstart), b.bin(tlen - tend), self.length.bin(tend - tstart)])

    @staticmethod
    def _survival(g, x):
        """S(x) = P(R > x) for the binned density g (EndProfile) at points x."""
        cdf = np.r_[0.0, np.cumsum(g.p)]
        return np.clip(1.0 - np.interp(x, g.edges, cdf), 0.0, 1.0)

    def _ends(self, feat, k):
        """(anchor-end bin, other-end bin) of component k."""
        return (feat[:, 1], feat[:, 0]) if self.anchors[k] == "3p" else (feat[:, 0], feat[:, 1])

    def _components(self, tx, feat):
        """(K, n) per-component densities P_k(d5, d3 | t)."""
        jr = feat[:, 2]
        L = self.tx_len[tx]
        out = np.empty((len(self.anchors), len(tx)))
        for k in range(len(self.anchors)):
            g = self.g[k]
            if self.anchors[k] == "internal":
                # mean of 1 / (L - r + 1) over r uniform in the bin's part below L
                lo = self.length.edges[jr]
                hi = np.maximum(np.minimum(self.length.edges[jr + 1], L), lo + 1e-9)
                inv = np.log(np.maximum(L - lo + 1.0, 1.0) / np.maximum(L - hi + 1.0, 1.0)) / (hi - lo)
                out[k] = g.weight(jr, tx, g.norm(self.Olen)) * np.maximum(inv, 1.0 / np.maximum(L + 1.0, 1.0))
                continue
            ja, jo = self._ends(feat, k)
            a = self.anchor[k]
            qa = a.weight(ja, tx, a.norm(self.O))
            full = jo < self.kf
            jf = np.minimum(jo, self.kf - 1)
            x = np.maximum(L - self.mid[ja] - self.delta, 0.0)
            other = np.where(full, self._survival(g, x) * self.e[k][jf] / self.width[jf], g.p[jr] / g.width[jr])
            out[k] = qa * other
        return out

    def weights(self, tx, feat):
        """c (n,) for alignments to transcripts tx with features feat."""
        w = self.pi @ self._components(tx, feat)
        u = self.uniform[tx]
        if u.any():
            lb = feat[u, 2]
            w[u] = self.length.p[lb] / self.length.width[lb] / (self.tx_len[tx[u]] + 1.0)
        return w

    def mstep(self, tx, feat, z, n_iter=200, tol=1e-10, pseudo=1e-3):
        u = self.uniform[tx]
        if u.any():
            g = np.bincount(feat[u, 2], weights=z[u], minlength=self.length.J) + pseudo
            self.length.p = g / g.sum()
        keep = ~u & (z > 0)
        tx, feat, z = tx[keep], feat[keep], z[keep]
        if not len(z):
            return
        P = self._components(tx, feat) * self.pi[:, None]
        rho = P / np.maximum(P.sum(0), 1e-300)
        if self.learn_pi:
            tot = rho @ z + pseudo
            self.pi = tot / tot.sum()
        T, L = len(self.tx_len), self.tx_len[tx]
        for k in range(len(self.anchors)):
            v = z * rho[k]
            if self.anchors[k] == "internal":
                g = self.g[k]
                g.mstep(np.bincount(feat[:, 2], weights=v, minlength=g.J), np.bincount(tx, weights=v, minlength=T),
                        self.Olen, pseudo=pseudo)
                continue
            ja, jo = self._ends(feat, k)
            a = self.anchor[k]
            a.mstep(np.bincount(ja, weights=v, minlength=a.J), np.bincount(tx, weights=v, minlength=T), self.O,
                    pseudo=pseudo)
            full = jo < self.kf
            e = np.bincount(jo[full], weights=v[full], minlength=self.kf) + pseudo
            self.e[k] = e / e.sum()
            # read lengths, right-censored at x = m - delta for full-length reads
            g = self.g[k]
            obs = np.bincount(feat[~full, 2], weights=v[~full], minlength=g.J)
            x = np.maximum(L[full] - self.mid[ja[full]] - self.delta, 0.0)
            xs, inv = np.unique(np.round(x), return_inverse=True)
            cw = np.bincount(inv, weights=v[full], minlength=len(xs))
            frac = np.clip((g.edges[1:][None, :] - xs[:, None]) / g.width[None, :], 0.0, 1.0)   # bin share above x
            p = g.p.copy()
            for _ in range(n_iter):
                S = np.maximum(frac @ p, 1e-300)
                new = obs + ((cw / S) @ frac) * p + pseudo
                new /= new.sum()
                done = np.abs(new - p).max() < tol
                p = new
                if done:
                    break
            g.p = p

    def full_length_fraction(self, lengths):
        """P(both read ends within delta of the transcript ends | L) at the given lengths."""
        L = np.asarray(lengths, dtype=np.float64)
        out = np.zeros(len(L))
        for k in range(len(self.anchors)):
            if self.anchors[k] == "internal":
                continue                                           # ~0 for L > delta
            a = self.anchor[k]
            O = a.overlap(L)[:, :self.kf]                          # anchor bins below delta
            qa = O * a.p[:self.kf] / a.norm(a.overlap(L))[:, None]
            S = self._survival(self.g[k], np.maximum(L[:, None] - self.mid[None, :self.kf] - self.delta, 0.0))
            out += self.pi[k] * (qa * S).sum(1)
        return out

    def profiles(self, lengths=(500, 1000, 2000, 3000, 5000, 8000, 12000)):
        """
        Long-format table of the fitted components: anchor profile p (untruncated bin masses; wide
        bins only long transcripts reach can hold much of it), full-length end, read length, pi,
        and the predicted full-length fraction at `lengths`.
        """
        import pandas as pd
        rows = [pd.DataFrame({"anchor": "all", "kind": "full_length_fraction", "x": np.asarray(lengths, dtype=np.float64),
                              "value": self.full_length_fraction(lengths)})]
        for k, end in enumerate(self.anchors):
            rows += [pd.DataFrame({"anchor": end, "kind": "anchor", "x": self.edges[:-1], "value": self.anchor[k].p}),
                     pd.DataFrame({"anchor": end, "kind": "full_end", "x": self.edges[:self.kf], "value": self.e[k]}),
                     pd.DataFrame({"anchor": end, "kind": "read_length", "x": self.g[k].edges[:-1], "value": self.g[k].p}),
                     pd.DataFrame({"anchor": end, "kind": "pi", "x": [0.0], "value": [self.pi[k]]})]
        return pd.concat(rows, ignore_index=True)


REL_EDGES = np.linspace(0.0, 1.0, 21)
D3_EDGES = np.array([0, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000,
                     5000, 10000, 30000], dtype=np.float64)


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
    b_t = L_t - r + 1. Targets in `uniform` (unspliced gene-span targets) get
    q = 1 / (L_t - r + 1) and are left out of the M-step (see TruncationModel).
    """

    def __init__(self, tx_len, read_len=150, mode="3p", edges=None, uniform=None):
        self.mode = mode
        self.uniform = np.zeros(len(tx_len), dtype=bool) if uniform is None else np.asarray(uniform, dtype=bool)
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
            w = self.prof.weight(j, tx, self.prof.norm(self.O))
        else:
            w = self.prof.p[j] / self.prof.width[j] / (self.span[tx] + 1.0)
        u = self.uniform[tx]
        w[u] = 1.0 / (self.span[tx[u]] + 1.0)
        return w

    def mstep(self, tx, feat, z):
        keep = ~self.uniform[tx]
        tx, feat, z = tx[keep], feat[keep], z[keep]
        A = np.bincount(feat[:, 0], weights=z, minlength=self.prof.J)
        if self.mode == "3p":
            self.prof.mstep(A, np.bincount(tx, weights=z, minlength=len(self.span)), self.O)
        else:
            p = A + 1e-3
            self.prof.p = p / p.sum()


FL_EDGES = np.r_[np.arange(0.0, 1000.0, 20.0), 1000.0, 1250.0, 1500.0, 2000.0, 3000.0]
FRAG_REL_EDGES = np.linspace(0.0, 1.0, 21)


class FragmentModel:
    """
    Paired-end short-read fragment model fitted jointly with the abundances.

    A fragment of length f starting at s (0-based) on transcript t of length
    L_t has relative start u = s / (L_t - f). Then
        P(read, t) proportional to mu_t fld(f) q(u),   b_t = sum_f fld(f) (L_t - f + 1)_+,
    with fld a fragment-length distribution over bins (density p_j / w_j) and q a
    positional coverage profile over bins of u (density q_k / v_k). So c = fld(f) q(u)
    and b is the effective length under fld (approximated at bin midpoints). A
    uniform q and salmon's fld give salmon's model without bias correction.

    M-step given expected reads z per alignment: q is the weighted histogram of u;
    fld maximises sum_j A_j log p_j - sum_l N_l log sum_j p_j K_lj with
    K_lj = sum_t mu_lt (L_t - m_j + 1)_+ (b depends on fld), solved by the fixed
    point p_j = A_j / sum_l N_l K_lj / B_l.
    """

    def __init__(self, tx_len, fl_edges=FL_EDGES, rel_edges=FRAG_REL_EDGES):
        self.tx_len = np.asarray(tx_len, dtype=np.float64)
        self.fe, self.re = np.asarray(fl_edges, dtype=np.float64), np.asarray(rel_edges, dtype=np.float64)
        self.fw, self.rw = np.diff(self.fe), np.diff(self.re)
        self.fmid = 0.5 * (self.fe[:-1] + self.fe[1:])
        self.pf = np.full(len(self.fw), 1.0 / len(self.fw))
        self.pq = np.full(len(self.rw), 1.0 / len(self.rw))
        self.O = np.maximum(self.tx_len[:, None] - self.fmid[None, :] + 1.0, 0.0)   # (T, J) start positions

    def init_lengths(self, flen, pseudo=1.0):
        """Start fld at the histogram of fragment lengths (e.g. of uniquely aligned fragments)."""
        h = np.bincount(self._fbin(flen), minlength=len(self.fw)) + pseudo
        self.pf = h / h.sum()

    def _fbin(self, flen):
        return np.clip(np.searchsorted(self.fe, np.asarray(flen, dtype=np.float64), side="right") - 1, 0, len(self.fw) - 1)

    def features(self, start, flen, tlen):
        """(n, 2): fragment-length bin and relative-start bin."""
        start, flen, tlen = (np.asarray(x, dtype=np.float64) for x in (start, flen, tlen))
        u = np.clip(start / np.maximum(tlen - flen, 1.0), 0.0, 1.0)
        ub = np.clip(np.searchsorted(self.re, u, side="right") - 1, 0, len(self.rw) - 1)
        return np.column_stack([self._fbin(flen), ub])

    def weights(self, tx, feat):
        fb, ub = feat[:, 0], feat[:, 1]
        return (self.pf[fb] / self.fw[fb]) * (self.pq[ub] / self.rw[ub])

    def rates(self):
        return np.maximum(self.O @ self.pf, 1e-12)

    def mstep(self, tx, feat, z, mus, Ns, n_iter=200, tol=1e-10, pseudo=1e-3):
        """z (n,) expected reads per alignment; mus, Ns per library sharing the model (molar mixtures, read totals)."""
        q = np.bincount(feat[:, 1], weights=z, minlength=len(self.rw)) + pseudo
        self.pq = q / q.sum()
        A = np.bincount(feat[:, 0], weights=z, minlength=len(self.fw)) + pseudo
        K = np.stack([m @ self.O for m in mus])               # (L, J)
        Ns = np.asarray(Ns, dtype=np.float64)
        p = self.pf.copy()
        for _ in range(n_iter):
            B = K @ p
            new = A / np.maximum((Ns / B) @ K, 1e-300)
            new /= new.sum()
            done = np.abs(new - p).max() < tol
            p = new
            if done:
                break
        self.pf = p


class LengthBias:
    """
    Library-specific sampling rate b_t = exp(x_t . w) multiplying the
    library's base rates, with x_t (P,) a piecewise-linear basis in log length
    (log L and hinges at the knots) and optional extra covariates, e.g. an
    indicator of unspliced targets whose rate scale differs (internal priming
    under poly(dT)). A read of transcript t has P(r, t) proportional to
    mu_t b_t q(r | t), so b also multiplies c. Columns are centred, so w = 0
    leaves the base rates unchanged up to a constant.

    b is confounded with theta unless theta is shared with another library
    whose rates are fixed (e.g. SR with effective lengths, or a reference LR
    protocol). The M-step maximises sum_t Z_t log b_t - N log sum_t mu_t b_t
    (concave in w) by Newton steps with a ridge penalty.

    Args:
        tx_len       : (T,) transcript lengths.
        knots        : hinge positions of the log-length basis.
        extra        : (T, P2) | None -- extra covariates appended to the basis.
        length_terms : bool -- include the log-length basis.
        ridge_extra  : float | None -- ridge on the extra covariates (default `ridge`), e.g. small
                       for a scale that the base rates only set roughly.
    """

    def __init__(self, tx_len, knots=(500, 1000, 2000, 4000, 8000), ridge=1.0, extra=None, length_terms=True,
                 ridge_extra=None):
        self.knots = tuple(knots)
        self.length_terms = length_terms
        cols = [self._basis(tx_len)] if length_terms else []
        if extra is not None:
            cols.append(np.asarray(extra, dtype=np.float64).reshape(len(tx_len), -1))
        X = np.column_stack(cols)
        self.center = X.mean(axis=0)
        self.X = X - self.center
        self.w = np.zeros(self.X.shape[1])
        P1 = self._basis(np.asarray(tx_len)[:1]).shape[1] if length_terms else 0
        self.ridge = np.full(self.X.shape[1], float(ridge))
        if ridge_extra is not None:
            self.ridge[P1:] = ridge_extra

    def _basis(self, lengths):
        logL = np.log(np.maximum(np.asarray(lengths, dtype=np.float64), 1.0))
        return np.column_stack([logL] + [np.maximum(0.0, logL - np.log(k)) for k in self.knots])

    def rates(self):
        return np.exp(self.X @ self.w)

    def rates_at(self, lengths):
        """b at arbitrary lengths from the length terms (extra covariates at their mean)."""
        if not self.length_terms:
            raise ValueError("no length terms")
        P = self._basis(lengths).shape[1]
        return np.exp((self._basis(lengths) - self.center[:P]) @ self.w[:P])

    def mstep(self, Z, mu, n_newton=30):
        """
        Z (T,) expected reads per transcript and mu (T,) molar mixture of one
        library, or lists of them for libraries sharing this length bias: the
        objective is sum_l [Z_l . log b - N_l log sum_t mu_lt b_t] - ridge |w|^2 / 2.
        Newton steps with backtracking (the Hessian can be nearly singular when
        mu is concentrated on a few targets).
        """
        Zs = Z if isinstance(Z, (list, tuple)) else [Z]
        mus = mu if isinstance(mu, (list, tuple)) else [mu]
        Ns = np.array([z.sum() for z in Zs])
        XZ = sum(self.X.T @ z for z in Zs)
        # libraries stacked (n_lib, T') over targets that some library can sample (mu > 0)
        M = np.stack([np.asarray(m_, dtype=np.float64) for m_ in mus])
        cols = np.flatnonzero((M > 0).any(axis=0))
        X = self.X[cols]
        logM = np.log(np.maximum(M[:, cols], 1e-300))
        P = len(self.w)

        def softmax(w):
            a = logM + (X @ w)[None, :]
            amax = a.max(axis=1, keepdims=True)
            e = np.exp(a - amax)
            s_ = e.sum(axis=1, keepdims=True)
            return e / s_, (amax + np.log(s_)).ravel()

        def objective(w):
            return XZ @ w - 0.5 * (self.ridge * w) @ w - Ns @ softmax(w)[1]

        f = objective(self.w)
        for _ in range(n_newton):
            p_, _ = softmax(self.w)
            m = p_ @ X                                         # (n_lib, P)
            grad = XZ - self.ridge * self.w - Ns @ m
            q = Ns @ p_                                        # (T',) sum_l N_l p_l
            H = np.diag(self.ridge) + (X * q[:, None]).T @ X - (m * Ns[:, None]).T @ m
            step = np.linalg.solve(H, grad)
            t = 1.0
            while True:
                w_new = self.w + t * step
                f_new = objective(w_new)
                if f_new >= f - 1e-12 * abs(f) or t < 1e-8:
                    break
                t *= 0.5
            done = np.abs(w_new - self.w).max() < 1e-8
            self.w, f = w_new, f_new
            if done:
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


class _KeyTables:
    """
    Unique (transcript, features) keys of the libraries in a bias fit. Bias-model
    weights and M-steps depend on a position only through its key, so they are
    evaluated per key: each library has a local table (int32 row per position),
    and each shared model a table over the union of its libraries' keys with a
    map from local rows to model rows.
    """

    def __init__(self, em, libs, groups):
        feats = [em.libs[l].feat for l in libs if em.libs[l].feat is not None]
        ncol = max((f.shape[1] for f in feats), default=0)
        self.width = np.ones(ncol, dtype=np.int64)
        for f in feats:
            for j in range(f.shape[1]):
                self.width[j] = max(self.width[j], int(f[:, j].max()) + 1 if len(f) else 1)
        if (self.width.prod() if ncol else 1) * (em.T_full + 1) >= 2 ** 62:
            raise ValueError("feature key space too large for int64 keys")
        self.ncol = {}
        self.loc = {}                       # l -> (row per position int32, unique keys, transcript per row)
        for l in libs:
            fl = em.libs[l]
            nc = 0 if fl.feat is None else fl.feat.shape[1]
            self.ncol[l] = nc
            idx, uk = pd.factorize(self._key(fl.tx, fl.feat, nc))
            self.loc[l] = (idx.astype(np.int32), np.asarray(uk, dtype=np.int64), self._decode(uk, nc)[0])
        self.model = {}                     # id(model) -> (tx_u, feat_u, {l: local row -> model row})
        for model, ls in groups:
            nc = self.ncol[ls[0]]
            if any(self.ncol[l] != nc for l in ls):
                raise ValueError("libraries sharing a bias model need the same feature columns")
            codes, uk = pd.factorize(np.concatenate([self.loc[l][1] for l in ls]))
            bounds = np.cumsum([0] + [len(self.loc[l][1]) for l in ls])
            maps = {l: codes[bounds[i]:bounds[i + 1]] for i, l in enumerate(ls)}
            tx_u, feat_u = self._decode(np.asarray(uk, dtype=np.int64), nc)
            self.model[id(model)] = (tx_u, feat_u, maps)

    def _key(self, tx, feat, nc):
        key = np.asarray(tx, dtype=np.int64).copy()
        for j in range(nc):
            key *= self.width[j]
            key += feat[:, j].astype(np.int64)
        return key

    def _decode(self, key, nc):
        key = np.asarray(key, dtype=np.int64).copy()
        cols = []
        for j in reversed(range(nc)):
            cols.append(key % self.width[j])
            key //= self.width[j]
        feat = np.column_stack(cols[::-1]).astype(np.int64) if nc else None
        return key, feat


def fit_bias(em, trunc=None, length=None, frag=None, n_outer=10, run_kw=None, verbose=False, inner_max_iter=500,
             init_theta=None, aggregate=True):
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
        frag   : dict {library index: FragmentModel} -- models that set both c (fld(f) q(u))
                 and b (effective length under fld); the library's base b is not used.
        n_outer: number of outer iterations.
        inner_max_iter : MM-step cap for the fits between bias updates (the final fit uses run_kw).
        init_theta : (K, T) | None -- warm start of the first fit (e.g. from a fit without bias models).
        aggregate : evaluate model weights and M-steps once per unique (transcript, features) key
                 (_KeyTables), with the per-position passes in the compiled kernels when available;
                 False evaluates them per position (reference implementation).
    Returns:
        JointResult of the final run; models are updated in place.
    """
    import time
    trunc, length, frag = trunc or {}, length or {}, frag or {}
    run_kw = dict(run_kw or {})
    libs = sorted(set(trunc) | set(length) | set(frag))
    base_c = {l: em.libs[l].c.copy() for l in libs}
    base_b = {l: em.libs[l].b.copy() for l in libs}
    groups_t, groups_f = list(_by_model(trunc).items()), list(_by_model(frag).items())
    tables = _KeyTables(em, libs, groups_t + groups_f) if aggregate else None
    rust = em.backend == "rust"
    threads = 1
    if rust:
        from . import native
        threads = em.threads or native.default_threads()
    pool = None
    if threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=threads)
    pmap = (lambda f, xs: list(pool.map(f, xs))) if pool else (lambda f, xs: [f(x) for x in xs])

    def apply_positions():
        for l in libs:
            fl = em.libs[l]
            c = base_c[l].copy()
            b = base_b[l].copy()
            if l in frag:
                c *= frag[l].weights(fl.tx, fl.feat)
                b = frag[l].rates()
            if l in trunc:
                # positional density times the base sampling rate (c = b q)
                c *= trunc[l].weights(fl.tx, fl.feat) * base_b[l][fl.tx]
            if l in length:
                r = length[l].rates()
                c *= r[fl.tx]
                b *= r
            em.set_bias(l, c=c, b=b)

    def apply_tables():
        wt = {id(m): m.weights(*tables.model[id(m)][:2]) for m, _ in groups_t + groups_f}
        rates = {id(m): m.rates() for m in set(length.values())}
        def one(l):
            idx, _, txl = tables.loc[l]
            wl = np.ones(len(txl))
            b = base_b[l].copy()
            if l in frag:
                wl *= wt[id(frag[l])][tables.model[id(frag[l])][2][l]]
                b = frag[l].rates()
            if l in trunc:
                wl *= wt[id(trunc[l])][tables.model[id(trunc[l])][2][l]] * base_b[l][txl]
            if l in length:
                r = rates[id(length[l])]
                wl *= r[txl]
                b = b * r
            if rust:
                c = native.gather_weights(base_c[l], idx, wl, threads=1)
            else:
                c = base_c[l] * wl[idx]
            return c, b
        for l, (c, b) in zip(libs, pmap(one, libs)):
            em.set_bias(l, c=c, b=b)

    def table_counts(theta, ls):
        """Expected reads per local key row of each library in ls (compiled scatter when available)."""
        th = em._compact(theta)
        def one(l):
            idx, uk, _ = tables.loc[l]
            fl = em.libs[l]
            if rust:
                return native.scatter_counts(fl.ec_ptr, fl.txc, fl.c, fl.counts, em.pi[l] @ th, idx, len(uk))
            return np.bincount(idx, weights=em.position_counts(theta, l), minlength=len(uk))
        return pmap(one, ls)

    def model_counts(model, ls, theta):
        tx_u, feat_u, maps = tables.model[id(model)]
        acc = np.zeros(len(tx_u))
        per_lib = table_counts(theta, ls)
        for l, a in zip(ls, per_lib):
            acc += np.bincount(maps[l], weights=a, minlength=len(tx_u))
        return tx_u, feat_u, acc, [a.sum() for a in per_lib]

    apply = apply_tables if aggregate else apply_positions

    # intermediate fits only need to move theta towards the optimum for the next bias update;
    # the final fit (after the last bias update) runs to convergence
    inner_kw = dict(run_kw, max_iter=min(run_kw.get("max_iter", 5000), inner_max_iter))
    t0 = time.time()
    apply()
    res = em.run(init_theta=init_theta, **inner_kw)
    t_run, t_m, t_apply = time.time() - t0, 0.0, 0.0
    for it in range(n_outer):
        t0 = time.time()
        # one M-step per model object, pooling the libraries that share it
        for model, ls in groups_t:
            if aggregate:
                tx_u, feat_u, acc, _ = model_counts(model, ls, res.theta)
                model.mstep(tx_u, feat_u, acc)
            else:
                fls = [em.libs[l] for l in ls]
                model.mstep(np.concatenate([fl.tx for fl in fls]), np.concatenate([fl.feat for fl in fls]),
                            np.concatenate([em.position_counts(res.theta, l) for l in ls]))
        for model, ls in groups_f:
            mus = [em.pi[l] @ res.theta for l in ls]
            if aggregate:
                tx_u, feat_u, acc, Ns = model_counts(model, ls, res.theta)
                model.mstep(tx_u, feat_u, acc, mus, Ns)
            else:
                fls = [em.libs[l] for l in ls]
                zs = [em.position_counts(res.theta, l) for l in ls]
                model.mstep(np.concatenate([fl.tx for fl in fls]), np.concatenate([fl.feat for fl in fls]),
                            np.concatenate(zs), mus, [z.sum() for z in zs])
        for model, ls in _by_model(length).items():
            mus = []
            for l in ls:
                ks = np.flatnonzero(em.pi[l] > 0)
                mus.append((em.pi[l, ks] @ res.theta[ks]) * base_b[l])
            model.mstep([res.expected_reads[l] for l in ls], mus)
        t1 = time.time()
        apply()
        t2 = time.time()
        # floor so transcripts zeroed by the previous run's final round can return
        kw = run_kw if it == n_outer - 1 else inner_kw
        res = em.run(init_theta=np.maximum(res.theta, 1e-10), **kw)
        t_m += t1 - t0
        t_apply += t2 - t1
        t_run += time.time() - t2
        if verbose:
            print(f"[bias] outer {it + 1}: loglik={res.loglik[-1]:.6e} n_iter={res.n_iter} "
                  f"(cumulative s: EM {t_run:.0f}, M-steps {t_m:.0f}, weights {t_apply:.0f})", flush=True)
    if pool:
        pool.shutdown()
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
