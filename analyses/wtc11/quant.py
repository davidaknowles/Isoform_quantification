"""
Quantify one WTC11 biosample from each platform alone and in combination.

Inputs (from align.sh): salmon SR runs {bs}_{nobias,bias} and minimap2
transcriptome PAFs {bs}_{pb,captrap,ont}.paf.gz. All libraries use the
GENCODE v38 transcript order of the reference fasta.

Fits (all one unit, no prior):
  SR:nobias, SR:bias                                 salmon ECs
  PB, CT, ONT (plain, +trunc)                        minimap2 ECs (best-score ties), truncation model
  PB@0.95, CT@0.95, ONT@0.95                         minimap2 ECs, alignments with AS >= 0.95 best
  SR:bias+PB+trunc, SR:bias+CT+trunc, SR:bias+ONT+trunc
Writes {out_dir}/{bs}_tpm.tsv.gz (transcripts x fits) and the learned
truncation profiles {bs}_trunc.tsv.

Run: python analyses/wtc11/quant.py <bs> <wtc11_dir> <ref_fasta> <out_dir> [minimap2_subdir] [lr_only]
    minimap2_subdir defaults to minimap2 (e.g. minimap2_sirv for a reference with spike-ins);
    lr_only skips the SR libraries.
"""

import sys
import time
from pathlib import Path

import pandas as pd

from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import TruncationModel, fit_with_truncation
from joli.paf import filter_alignments, library_from_alignments, read_fasta_lengths, read_paf
from joli.salmon_io import library_from_salmon


def main(bs, wtc11_dir, ref_fasta, out_dir, mm_subdir="minimap2", lr_only=False):
    wtc11_dir, out_dir = Path(wtc11_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, tx_len = read_fasta_lengths(ref_fasta)
    t0 = time.time()
    sr = {} if lr_only else {
        m: library_from_salmon(f"SR:{m}", wtc11_dir / "salmon" / f"{bs}_{m}", transcript_names=names)
        for m in ["nobias", "bias"]}
    print(f"loaded SR ({time.time() - t0:.0f} s)", flush=True)
    lr = {}
    for plat, strand in [("PB", None), ("CT", None), ("ONT", "+")]:
        t0 = time.time()
        f = {"PB": "pb", "CT": "captrap", "ONT": "ont"}[plat]
        raw = read_paf(wtc11_dir / mm_subdir / f"{bs}_{f}.paf.gz")
        aln = filter_alignments(raw, strand=strand)
        lr[plat + "@0.95"] = library_from_alignments(plat, filter_alignments(raw, score_frac=0.95, strand=strand), names)
        del raw
        feat = TruncationModel(tx_len).features(aln["tstart"], aln["tend"], aln["tlen"])
        lr[plat] = library_from_alignments(plat, aln, names)
        lr[plat + "+trunc"] = library_from_alignments(plat, aln, names, features=feat)
        print(f"loaded {plat}: {aln['read'].nunique()} reads, {len(aln)} alignments ({time.time() - t0:.0f} s)", flush=True)

    fits = {} if lr_only else {"SR:nobias": [sr["nobias"]], "SR:bias": [sr["bias"]]}
    for plat in ["PB", "CT", "ONT"]:
        fits[plat] = [lr[plat]]
        fits[plat + "@0.95"] = [lr[plat + "@0.95"]]
        fits[plat + "+trunc"] = [lr[plat + "+trunc"]]
        if not lr_only:
            fits[f"SR:bias+{plat}+trunc"] = [sr["bias"], lr[plat + "+trunc"]]
    tpm, profiles, summary = {}, [], []
    for fit, libs in fits.items():
        t0 = time.time()
        em = JointEM(libs, units=[0] * len(libs))
        trunc = {l: TruncationModel(tx_len) for l, lib in enumerate(libs) if lib.feat is not None}
        res = fit_with_truncation(em, trunc, n_outer=8) if trunc else em.run()
        tpm[fit] = molar_to_tpm(res.theta[0])
        for l, m in trunc.items():
            for end, prof in [("5p", m.end5), ("3p", m.end3)]:
                profiles.append(pd.DataFrame({"fit": fit, "library": libs[l].name, "end": end,
                                              "bin_lo": prof.edges[:-1], "bin_hi": prof.edges[1:], "p": prof.p}))
        summary.append({"fit": fit, "loglik": res.loglik[-1], "n_iter": res.n_iter,
                        "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)
    pd.DataFrame(tpm, index=names).to_csv(out_dir / f"{bs}_tpm.tsv.gz", sep="\t", float_format="%.6g")
    pd.concat(profiles).to_csv(out_dir / f"{bs}_trunc.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(out_dir / f"{bs}_fits.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:6], lr_only=len(sys.argv) > 6 and sys.argv[6] == "lr_only")
