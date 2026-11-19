import numpy as np

from joli.deconv import mixture_weights


def test_mixture_weights_recovers_composition():
    rng = np.random.default_rng(0)
    K, G = 4, 3000
    S = rng.dirichlet(np.full(G, 0.3), size=K)
    w = np.array([0.6, 0.25, 0.15, 0.0])
    y = rng.multinomial(2_000_000, w @ S)
    est, ll = mixture_weights(y, S)
    np.testing.assert_allclose(est.sum(), 1.0)
    np.testing.assert_allclose(est, w, atol=0.01)
    assert ll > float(y[y > 0] @ np.log((np.full(K, 1 / K) @ S)[y > 0]))


def test_single_gene_counts():
    from joli.deconv import single_gene_counts
    gene_of = np.array([0, 0, 1, -1])
    ptr = np.array([0, 2, 3, 5, 6])                 # ECs {0,1} gene 0; {2} gene 1; {1,2} two genes; {3} none
    values = np.array([0, 1, 2, 1, 2, 3])
    np.testing.assert_allclose(single_gene_counts(ptr, values, np.array([5.0, 2, 7, 1]), gene_of, 2), [5.0, 2.0])
