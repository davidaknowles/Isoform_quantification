"""
cells.py
========
Cell calling helpers for single-cell libraries.
"""

import numpy as np


def knee_threshold(counts, min_count=10, grid=200, smooth=5):
    """
    Count threshold at the inflection of the barcode-rank curve, as in
    DropletUtils::barcodeRanks: the steepest descent of log count against
    log rank, on an even grid of log ranks smoothed by a running mean.

    Args:
        counts    : array (n_barcodes,) UMIs per barcode.
        min_count : barcodes below this are ignored.
    Returns the count at the inflection (keep barcodes with counts >= it).
    """
    c = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    c = c[c >= min_count]
    if len(c) < 10:
        return float(c[-1]) if len(c) else float(min_count)
    x = np.log10(np.arange(1, len(c) + 1))
    y = np.log10(c)
    xg = np.linspace(x[0], x[-1], grid)
    yg = np.interp(xg, x, y)
    yg = np.convolve(yg, np.ones(smooth) / smooth, mode="same")
    slope = np.gradient(yg, xg)
    lo, hi = smooth, grid - smooth                       # ignore edge effects of the smoother
    k = lo + int(np.argmin(slope[lo:hi]))
    return float(10 ** np.interp(xg[k], x, y))
