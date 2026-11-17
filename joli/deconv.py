"""
Gene-level deconvolution of bulk libraries into reference profiles.

A bulk library with gene counts y (G,) is modelled as multinomial with probabilities sum_k w_k s_kg, where
s_k (G,) are reference gene fractions (e.g. per cell type, from short reads) and w (K,) mixture weights on
the simplex: the joint model's mixture likelihood aggregated to genes. Weights are fitted by EM
(multiplicative updates w_k <- w_k sum_g y_g s_kg / mu_g / sum_g y_g), which increases the likelihood
monotonically and keeps w on the simplex.
"""

import numpy as np


def mixture_weights(y, S, w0=None, max_iter=5000, tol=1e-10):
    """
    y : (G,) nonnegative counts (or count-scaled fractions) of the bulk library.
    S : (K, G) reference profiles; rows are normalised to sum to 1.
    Returns (w (K,), loglik) with loglik = sum_g y_g log sum_k w_k s_kg.
    """
    y = np.asarray(y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    S = S / S.sum(axis=1, keepdims=True)
    keep = (y > 0) & (S.sum(axis=0) > 0)
    y, S = y[keep], S[:, keep]
    K = S.shape[0]
    w = np.full(K, 1.0 / K) if w0 is None else np.asarray(w0, dtype=np.float64) / np.sum(w0)
    n = y.sum()
    ll_old = -np.inf
    for _ in range(max_iter):
        mu = w @ S
        ll = float(y @ np.log(mu))
        w = w * (S @ (y / mu)) / n
        if ll - ll_old <= tol * abs(ll):
            break
        ll_old = ll
    return w, float(y @ np.log(w @ S))
