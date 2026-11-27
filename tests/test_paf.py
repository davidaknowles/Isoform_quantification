

def test_score_weights_bins_and_weights():
    import numpy as np
    import pandas as pd
    from joli.paf import filter_alignments, score_weights
    aln = pd.DataFrame({"read": [0, 0, 0, 1, 1], "tx": ["a", "b", "c", "a", "b"], "AS": [100, 97, 90, 50, 50],
                        "strand": "+", "qstart": 0, "qend": 10, "qlen": 10})
    a = filter_alignments(aln, score_frac=0.95)
    assert set(zip(a["read"], a["tx"])) == {(0, "a"), (0, "b"), (1, "a"), (1, "b")}
    b, w = score_weights(a, D=5.0)
    gap = a["AS_gap"].to_numpy()
    np.testing.assert_allclose(w, np.exp(-gap / 5.0))
    assert b[gap == 0].tolist() == [0, 0, 0] and b[gap == 3].tolist() == [2]
