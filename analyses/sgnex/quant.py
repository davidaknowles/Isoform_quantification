"""
Quantify one SG-NEx replicate (cell line x replicate) from Illumina (salmon)
and ONT direct cDNA and PCR-cDNA (minimap2), alone and jointly.

Fits (one unit each, no prior):
  SR:nobias, SR:bias                       salmon ECs
  {LR}, {LR}+trunc                         each long-read protocol
  SR+{LR}+trunc                            SR (bias-corrected) with one protocol
  SR+{LR}+trunc+len                        ... plus a length-bias term for the LR library
  SR+allLR+trunc+len                       SR with all protocols, each LR with truncation and length bias
  SRfrag, SRfrag+{LR}+trunc, SRfrag+allLR+trunc
                                           SR from salmon fragment alignments with a fragment-length
                                           distribution and positional coverage profile fitted jointly
                                           (FragmentModel), no salmon bias correction
SR effective lengths are the fixed reference that makes LR length bias identifiable.

Writes {out_dir}/{cell}_rep{rep}_tpm.tsv.gz, _fits.tsv and _bias.tsv.

Run: python analyses/sgnex/quant.py <cell> <rep> <sgnex_dir> <out_dir>
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import FragmentModel, LengthBias, TruncationModel, fit_bias
from joli.paf import filter_alignments, library_from_alignments, read_fasta_lengths, read_paf
from joli.salmon_io import library_from_fragments, library_from_salmon, read_fragments

# ONT direct RNA is left out: too low throughput to be a useful long-read input
LR_PROTOCOLS = {"directcDNA": None, "cDNA": None}


def main(cell, rep, sgnex_dir, out_dir):
    sgnex_dir, out_dir = Path(sgnex_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cell}_rep{rep}"
    names, tx_len = read_fasta_lengths(sgnex_dir / "ref" / "hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa")
    sr = {m: library_from_salmon(f"SR:{m}", sgnex_dir / "salmon" / f"{stem}_Illumina_{m}", transcript_names=names)
          for m in ["nobias", "bias"]}
    # SR fragments (salmon --writeMappings) for the jointly fitted fragment model
    fdir = sgnex_dir / "sr_fragments" / f"{stem}_Illumina"
    srf, frag_init = None, None
    if (fdir / "fragments.tsv.gz").exists():
        t0 = time.time()
        frag_init = FragmentModel(tx_len)
        frags = read_fragments(fdir / "fragments.tsv.gz", fdir / "refs.txt", names)
        srf = library_from_fragments("SR:frag", frags, names, tx_len, frag_init)
        print(f"loaded SR fragments: {frags['read'].nunique()} fragments, {len(frags)} alignments ({time.time() - t0:.0f} s)", flush=True)
        del frags
    lr = {}
    for proto, strand in LR_PROTOCOLS.items():
        paf = sgnex_dir / "minimap2" / f"{stem}_{proto}.paf.gz"
        if not paf.exists():
            continue
        t0 = time.time()
        aln = filter_alignments(read_paf(paf), strand=strand)
        feat = TruncationModel(tx_len).features(aln["tstart"], aln["tend"], aln["tlen"])
        lr[proto] = library_from_alignments(proto, aln, names, features=feat)
        print(f"loaded {proto}: {aln['read'].nunique()} reads ({time.time() - t0:.0f} s)", flush=True)

    # (libraries, truncate LR?, length-bias LR?, fragment model for SR:frag?)
    fits = {"SR:nobias": ([sr["nobias"]], False, False, False), "SR:bias": ([sr["bias"]], False, False, False)}
    for proto, lib in lr.items():
        fits[proto] = ([lib], False, False, False)
        fits[f"{proto}+trunc"] = ([lib], True, False, False)
        fits[f"SR+{proto}+trunc"] = ([sr["bias"], lib], True, False, False)
        fits[f"SR+{proto}+trunc+len"] = ([sr["bias"], lib], True, True, False)
    fits["SR+allLR+trunc+len"] = ([sr["bias"]] + list(lr.values()), True, True, False)
    if srf is not None:
        fits["SRfrag"] = ([srf], False, False, True)
        for proto, lib in lr.items():
            fits[f"SRfrag+{proto}+trunc"] = ([srf, lib], True, False, True)
        fits["SRfrag+allLR+trunc"] = ([srf] + list(lr.values()), True, False, True)

    tpm, summary, bias = {}, [], []
    for fit, (libs, do_trunc, do_len, do_frag) in fits.items():
        t0 = time.time()
        em = JointEM(libs, units=[0] * len(libs))
        lr_idx = [l for l, lib in enumerate(libs) if lib.platform == "long"]
        trunc = {l: TruncationModel(tx_len) for l in lr_idx} if do_trunc else {}
        length = {l: LengthBias(tx_len) for l in lr_idx} if do_len else {}
        frag = {}
        if do_frag:
            fm = FragmentModel(tx_len)
            fm.pf = frag_init.pf.copy()
            frag = {l: fm for l, lib in enumerate(libs) if lib.name == "SR:frag"}
        if trunc or length or frag:
            res = fit_bias(em, trunc=trunc, length=length, frag=frag, n_outer=8)
        else:
            # plain fits ignore the truncation features (c = 1)
            for l in lr_idx:
                em.set_bias(l, c=np.ones_like(em.libs[l].c))
            res = em.run()
        tpm[fit] = molar_to_tpm(res.theta[0])
        for m in {id(m): m for m in frag.values()}.values():
            bias.append(pd.DataFrame({"fit": fit, "library": "SR:frag", "kind": "fld", "x": m.fe[:-1], "value": m.pf}))
            bias.append(pd.DataFrame({"fit": fit, "library": "SR:frag", "kind": "sr_position", "x": m.re[:-1], "value": m.pq}))
        for l, m in trunc.items():
            for end, prof in [("5p", m.end5), ("3p", m.end3)]:
                bias.append(pd.DataFrame({"fit": fit, "library": libs[l].name, "kind": f"trunc_{end}",
                                          "x": prof.edges[:-1], "value": prof.p}))
        for l, m in length.items():
            grid = np.array([300, 500, 1000, 2000, 4000, 8000, 16000])
            bias.append(pd.DataFrame({"fit": fit, "library": libs[l].name, "kind": "length",
                                      "x": grid, "value": m.rates_at(grid)}))
        summary.append({"fit": fit, "loglik": res.loglik[-1], "n_iter": res.n_iter,
                        "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)
    pd.DataFrame(tpm, index=names).to_csv(out_dir / f"{stem}_tpm.tsv.gz", sep="\t", float_format="%.6g")
    pd.DataFrame(summary).to_csv(out_dir / f"{stem}_fits.tsv", sep="\t", index=False)
    pd.concat(bias).to_csv(out_dir / f"{stem}_bias.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:5])
