"""Tests for alevin-fry pseudobulk loading."""

import gzip

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from joli.alevin import library_from_pseudobulk, primer_rates, pseudobulk_counts


def write_fry(d, barcodes, features, counts, ecs):
    a = d / "alevin"
    a.mkdir(parents=True)
    (a / "quants_mat_rows.txt").write_text("\n".join(barcodes) + "\n")
    (a / "quants_mat_cols.txt").write_text("\n".join(features) + "\n")
    scipy.io.mmwrite(a / "geqc_counts.mtx", sp.csr_matrix(np.asarray(counts, dtype=float)))
    with gzip.open(a / "gene_eqclass.txt.gz", "wt") as fh:
        fh.write(f"{len(features)}\n{len(ecs)}\n")
        for i, e in enumerate(ecs):
            fh.write(" ".join(map(str, e)) + f" {i}\n")


def test_pseudobulk_pools_ecs_across_libraries(tmp_path):
    feats = ["t1", "t2", "g1-U"]
    # library A: EC0 = {t1}, EC1 = {t1, t2}; library B lists the same sets in the other order
    write_fry(tmp_path / "A", ["a1", "a2"], feats, [[3, 1], [0, 2]], [[0], [0, 1]])
    write_fry(tmp_path / "B", ["b1"], feats, [[4, 5]], [[0, 1], [0]])
    cells = pd.DataFrame({"barcode": ["a1", "a2", "b1"], "library": ["A", "A", "B"],
                          "subject": ["s", "s", "s"], "cell_type": ["T", "T", "T"],
                          "primer": ["poly(dT)", "random hexamer", "poly(dT)"]})
    features, ec_keys, groups, counts = pseudobulk_counts({"A": tmp_path / "A", "B": tmp_path / "B"}, cells)
    tot = {k: counts[groups.index[groups.primer == "poly(dT)"][0], e] for e, k in enumerate(ec_keys)}
    assert tot[("t1",)] == 3 + 5 and tot[("t1", "t2")] == 1 + 4
    assert groups.set_index("primer").loc["poly(dT)", "n_cells"] == 2
    eff = np.array([100.0, 200.0, 1000.0])
    lib = library_from_pseudobulk("x", ec_keys, counts[0], features, primer_rates("poly(dT)", eff))
    assert lib.tcc.ec_counts.sum() == counts[0].sum()
    np.testing.assert_array_equal(primer_rates("random hexamer", eff), eff)
