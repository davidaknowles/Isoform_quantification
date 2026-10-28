"""Tests for RAD dump helpers and the short-read position model."""

import numpy as np
import pandas as pd

from joli.joint import JointEM
from joli.lr_bias import ReadPositionModel, fit_bias
from joli.paf import library_from_alignments
from joli.rad import decode_2bit


def test_decode_2bit():
    # A=0 C=1 G=2 T=3, first base in the highest bits
    assert decode_2bit([0b00011011, 0b11100100], 4) == ["ACGT", "TGCA"]


def test_position_model_resolves_alternative_polya():
    """
    Genes with a short isoform [0, S) and a 3'-extended isoform [0, L). 150-nt
    reads sit at d3 ~ Exponential(300) from the 3' end of their molecule's
    isoform, as for poly(dT) priming. Reads inside [0, S) are compatible with
    both; the position model should assign them by distance to each 3' end.
    """
    rng = np.random.default_rng(2)
    n_genes, r = 150, 150
    names, lens = [], []
    for g in range(n_genes):
        S = rng.integers(1000, 2500)
        L = S + rng.integers(500, 2000)
        names += [f"g{g}_short", f"g{g}_long"]
        lens += [S, L]
    lens = np.array(lens)
    theta = rng.dirichlet(np.ones(2 * n_genes))
    n = 60000
    tx = rng.choice(2 * n_genes, size=n, p=theta)
    d3 = np.minimum(rng.exponential(300, size=n).astype(int), lens[tx] - r)
    start = lens[tx] - r - d3
    rows = []
    for m, (t, s) in enumerate(zip(tx, start)):
        g = t // 2
        S, L = lens[2 * g], lens[2 * g + 1]
        rows.append((m, f"g{g}_long", s, s + r, L))
        if s + r <= S:
            rows.append((m, f"g{g}_short", s, s + r, S))
    aln = pd.DataFrame(rows, columns=["read", "tx", "start", "end", "tlen"])
    b = np.ones(len(names))           # poly(dT): one UMI per molecule
    plain = JointEM([library_from_alignments("x", aln, names, b=b)]).run()
    model = ReadPositionModel(lens, read_len=r, mode="3p")
    lib = library_from_alignments("x", aln, names, features=model.features(aln.start, aln.end, aln.tlen), b=b)
    fitted = fit_bias(JointEM([lib]), trunc={0: model}, n_outer=10)
    err0 = np.abs(plain.theta[0] - theta).sum()
    err1 = np.abs(fitted.theta[0] - theta).sum()
    assert err1 < 0.5 * err0, (err0, err1)
    # learned profile puts most mass within 500 nt of the 3' end
    assert model.prof.p[model.prof.edges[:-1] < 500].sum() > 0.7


def test_encode_roundtrip():
    from joli.rad import encode_2bit
    s = ["ACGTACGTACGTACGTACGTACGT", "TTTTGGGGCCCCAAAAACGTACGT"]
    assert decode_2bit(encode_2bit(s, 24), 24) == s


def test_knee_threshold():
    from joli.cells import knee_threshold
    rng = np.random.default_rng(0)
    counts = np.r_[rng.lognormal(np.log(3000), 0.5, 2000), rng.lognormal(np.log(30), 0.7, 50000)]
    t = knee_threshold(counts)
    assert 150 < t < 1500
