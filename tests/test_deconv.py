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
