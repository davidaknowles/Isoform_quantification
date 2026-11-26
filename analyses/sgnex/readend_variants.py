"""
Long-read-only ablation of the read-end model against oarfish's truncation model, on one SG-NEx replicate.

Per ONT protocol (direct cDNA, PCR-cDNA), LR alone with the AnchoredTruncationModel fitted jointly (fit_bias):
  anch           best-score ties (filter_alignments default), 3'/5'-anchored components (current default)
  anch+int       plus the uniform ("internal") component, as oarfish's uniform placement
  anch+asD<D>    alignments with AS >= 0.95 * best, weighted exp(-(best AS - AS) / D) (oarfish's score probability);
                 the AS gap bin is an extra feature, so reads in an equivalence class share their gaps
  anch+int+asD10, anch+int+asD10all (all minimap2 secondary alignments, no score filter)
Writes {out_dir}/{cell}_rep{rep}_tpm.tsv.gz (columns <protocol>+<variant>) for spikein_eval.py.

Run: python analyses/sgnex/readend_variants.py <cell> <rep> <sgnex_dir> <out_dir>
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import AnchoredTruncationModel, fit_bias
from joli.paf import filter_alignments, library_from_alignments, read_fasta_lengths, read_paf

GAP_EDGES = np.array([0, 1, 3, 6, 10, 15, 20, 30, 50, 1e9])
VARIANTS = {"anch": (1.0, None, ("3p", "5p")), "anch+int": (1.0, None, ("3p", "5p", "internal")),
            "anch+asD5": (0.95, 5.0, ("3p", "5p")), "anch+asD10": (0.95, 10.0, ("3p", "5p")),
            "anch+int+asD10": (0.95, 10.0, ("3p", "5p", "internal")),
            "anch+int+asD10all": (0.0, 10.0, ("3p", "5p", "internal"))}


def main(cell, rep, sgnex_dir, out_dir):
    sgnex_dir, out = Path(sgnex_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cell}_rep{rep}"
    names, tx_len = read_fasta_lengths(sgnex_dir / "ref" / "hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa")
    tpm, rows = {}, []
    for proto in ["directcDNA", "cDNA"]:
        raw = read_paf(sgnex_dir / "minimap2" / f"{stem}_{proto}.paf.gz")
        cache = {}
        for v, (frac, D, anchors) in VARIANTS.items():
            t0 = time.time()
            if frac not in cache:
                cache[frac] = filter_alignments(raw, score_frac=frac)
            aln = cache[frac]
            model = AnchoredTruncationModel(tx_len, anchors=anchors)
            feat = model.features(aln["tstart"], aln["tend"], aln["tlen"])
            weight = None
            if D is not None:
                gap = aln["AS_gap"].to_numpy(dtype=np.float64)
                feat = np.column_stack([feat, np.searchsorted(GAP_EDGES, gap, side="right") - 1])
                weight = np.exp(-gap / D)
            lib = library_from_alignments(proto, aln, names, features=feat, weight=weight)
            em = JointEM([lib], units=[0])
            res = fit_bias(em, trunc={0: model}, n_outer=8)
            tpm[f"{proto}+{v}"] = molar_to_tpm(res.theta[0])
            rows.append({"fit": f"{proto}+{v}", "alignments": len(aln), "pi": np.round(model.pi, 3).tolist(),
                         "n_iter": res.n_iter, "seconds": round(time.time() - t0)})
            print(rows[-1], flush=True)
        del raw, cache
    pd.DataFrame(tpm, index=names).to_csv(out / f"{stem}_tpm.tsv.gz", sep="\t", float_format="%.6g")
    pd.DataFrame(rows).to_csv(out / f"{stem}_fits.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:5])
