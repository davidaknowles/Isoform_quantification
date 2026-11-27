"""
Long-read-only ablation of the read-end model against oarfish's truncation model, on one SG-NEx replicate.

Per ONT protocol (direct cDNA, PCR-cDNA), LR alone with the AnchoredTruncationModel fitted jointly (fit_bias):
  anch           best-score ties (filter_alignments default), 3'/5'-anchored components (current default)
  anch+int       plus the uniform ("internal") component, as oarfish's uniform placement
  anch+asD<D>    alignments with AS >= 0.95 * best, weighted exp(-(best AS - AS) / D) (oarfish's score probability);
                 the AS gap bin is an extra feature, so reads in an equivalence class share their gaps
  anch+int+asD10, anch+int+asD10all (all minimap2 secondary alignments, no score filter)
With a comma-separated variant list only those run; a variant starting with "joint:" (e.g. joint:anch+asD5) fits
SR fragments (FragmentModel) with both ONT protocols, each with its own read-end model (as quant.py SRfrag+allLR+anch),
column SRfrag+allLR+<variant>.
Writes {out_dir}/{cell}_rep{rep}_tpm.tsv.gz (columns <protocol>+<variant>) for spikein_eval.py.

Run: python analyses/sgnex/readend_variants.py <cell> <rep> <sgnex_dir> <out_dir> [variants]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import AnchoredTruncationModel, FragmentModel, fit_bias
from joli.paf import filter_alignments, library_from_alignments, read_fasta_lengths, read_paf, score_weights
from joli.salmon_io import library_from_fragments, read_fragments

VARIANTS = {"anch": (1.0, None, ("3p", "5p")), "anch+int": (1.0, None, ("3p", "5p", "internal")),
            "anch+asD5": (0.95, 5.0, ("3p", "5p")), "anch+asD10": (0.95, 10.0, ("3p", "5p")),
            "anch+int+asD10": (0.95, 10.0, ("3p", "5p", "internal")),
            "anch+int+asD10all": (0.0, 10.0, ("3p", "5p", "internal")),
            "anch+asD3": (0.95, 3.0, ("3p", "5p")), "anch+asD2": (0.95, 2.0, ("3p", "5p"))}


def lr_library(raw_by_frac, raw, proto, names, tx_len, v):
    frac, D, anchors = VARIANTS[v]
    if frac not in raw_by_frac:
        raw_by_frac[frac] = filter_alignments(raw, score_frac=frac)
    aln = raw_by_frac[frac]
    model = AnchoredTruncationModel(tx_len, anchors=anchors)
    feat = model.features(aln["tstart"], aln["tend"], aln["tlen"])
    weight = None
    if D is not None:
        gap_bin, weight = score_weights(aln, D)
        feat = np.column_stack([feat, gap_bin])
    return library_from_alignments(proto, aln, names, features=feat, weight=weight), model, len(aln)


def main(cell, rep, sgnex_dir, out_dir, variants=None):
    sgnex_dir, out = Path(sgnex_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cell}_rep{rep}"
    names, tx_len = read_fasta_lengths(sgnex_dir / "ref" / "hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa")
    tpm, rows = {}, []
    wanted = list(VARIANTS) if variants is None else variants.split(",")
    lr_only = [v for v in wanted if not v.startswith("joint:")]
    joint = [v.split(":", 1)[1] for v in wanted if v.startswith("joint:")]
    joint_libs = {v: [] for v in joint}
    for proto in ["directcDNA", "cDNA"]:
        raw = read_paf(sgnex_dir / "minimap2" / f"{stem}_{proto}.paf.gz")
        cache = {}
        for v in lr_only:
            t0 = time.time()
            lib, model, n_aln = lr_library(cache, raw, proto, names, tx_len, v)
            em = JointEM([lib], units=[0])
            res = fit_bias(em, trunc={0: model}, n_outer=8)
            tpm[f"{proto}+{v}"] = molar_to_tpm(res.theta[0])
            rows.append({"fit": f"{proto}+{v}", "alignments": n_aln, "pi": np.round(model.pi, 3).tolist(),
                         "n_iter": res.n_iter, "seconds": round(time.time() - t0)})
            print(rows[-1], flush=True)
        for v in joint:
            joint_libs[v].append(lr_library(cache, raw, proto, names, tx_len, v)[:2])
        del raw, cache
    if joint:
        fdir = sgnex_dir / "sr_fragments" / f"{stem}_Illumina"
        frag_init = FragmentModel(tx_len)
        frags = read_fragments(fdir / "fragments.tsv.gz", fdir / "refs.txt", names)
        srf = library_from_fragments("SR:frag", frags, names, tx_len, frag_init)
        del frags
        for v in joint:
            t0 = time.time()
            libs = [srf] + [lib for lib, _ in joint_libs[v]]
            em = JointEM(libs, units=[0] * len(libs))
            fm = FragmentModel(tx_len)
            fm.pf = frag_init.pf.copy()
            res = fit_bias(em, trunc={1 + i: m for i, (_, m) in enumerate(joint_libs[v])}, frag={0: fm}, n_outer=8)
            tpm[f"SRfrag+allLR+{v}"] = molar_to_tpm(res.theta[0])
            rows.append({"fit": f"SRfrag+allLR+{v}", "n_iter": res.n_iter, "seconds": round(time.time() - t0)})
            print(rows[-1], flush=True)
    pd.DataFrame(tpm, index=names).to_csv(out / f"{stem}_tpm.tsv.gz", sep="\t", float_format="%.6g")
    pd.DataFrame(rows).to_csv(out / f"{stem}_fits.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:6])
