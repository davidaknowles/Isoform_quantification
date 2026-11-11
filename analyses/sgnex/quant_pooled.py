"""
Spike-in test of platform pooling and the design prior on SG-NEx: all replicates fitted together.

Libraries per replicate (H9 rep2-4: SIRV-4; HEYA8 rep1-3: sequins Mix A v2 and SIRV E2): Illumina
fragments (SR, FragmentModel) and ONT direct cDNA and PCR-cDNA (minimap2, AnchoredTruncationModel),
each library with its own bias model as in quant.py. Fits:
  LR         both ONT protocols, one abundance vector per replicate
  joint      SR + both ONT protocols, one abundance vector per replicate (= SRfrag+allLR+anch of quant.py)
  tpool      units (replicate, protocol) shrunk towards a shared replicate profile (transcript level, tau 1)
  design     units (replicate, protocol): log abundance = intercept + protocol (flat) + cell line (sd 2)
             + replicate (sd REP_SD) + unit noise (tau 1)
  LR_design  the same design prior on the two ONT protocols alone (control for what SR adds)
  gene       units (replicate, protocol) sharing within-gene isoform fractions, with protocol-specific
             gene totals (JointEM tie/groups)
  gpool_t<tau>  as gene, with protocol gene totals shrunk towards the replicate's (log-scale sd tau)
Pooled fits start from the previous fit's abundances and bias models. Per replicate, the pooled fits
report each protocol's unit (<fit>@<protocol>) and a consensus (<fit>): the shared profile for tpool,
the design profile with protocol effects averaged for design and LR_design, and the geometric mean of
the protocol units for gene and gpool (shared within-gene fractions times geometric-mean gene totals).
An optional comma-separated list of fits runs only those (plus their warm starts) and adds their
columns to existing output files.

Writes {out_dir}/{cell}_rep{rep}_tpm.tsv.gz (scored by spikein_eval.py) and fits.tsv.

Run: python analyses/sgnex/quant_pooled.py <sgnex_dir> <out_dir> [fits]
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joli.joint import JointEM, molar_to_tpm
from joli.lr_bias import AnchoredTruncationModel, FragmentModel, fit_bias
from joli.paf import filter_alignments, library_from_alignments, read_fasta_lengths, read_paf
from joli.salmon_io import library_from_fragments, read_fragments

REPLICATES = [("H9", 2), ("H9", 3), ("H9", 4), ("HEYA8", 1), ("HEYA8", 2), ("HEYA8", 3)]
PROTOCOLS = ["SR", "directcDNA", "cDNA"]
LR_STRAND = {"directcDNA": None, "cDNA": None}
TAU = 1.0
REP_SD = 0.5
CELL_SD = 2.0
RUN_KW = dict(max_iter=20000, ll_rel_tol=1e-10)


def load(sgnex_dir, names, tx_len):
    """Libraries keyed by (stem, protocol) and the initial fragment-length distributions of the SR libraries."""
    libs, frag_init = {}, {}
    for cell, rep in REPLICATES:
        stem = f"{cell}_rep{rep}"
        t0 = time.time()
        fdir = sgnex_dir / "sr_fragments" / f"{stem}_Illumina"
        fm = FragmentModel(tx_len)
        frags = read_fragments(fdir / "fragments.tsv.gz", fdir / "refs.txt", names)
        libs[(stem, "SR")] = library_from_fragments(f"{stem}|SR", frags, names, tx_len, fm)
        frag_init[stem] = fm.pf.copy()
        del frags
        for proto, strand in LR_STRAND.items():
            aln = filter_alignments(read_paf(sgnex_dir / "minimap2" / f"{stem}_{proto}.paf.gz"), strand=strand)
            feat = AnchoredTruncationModel(tx_len).features(aln["tstart"], aln["tend"], aln["tlen"])
            libs[(stem, proto)] = library_from_alignments(f"{stem}|{proto}", aln, names, features=feat)
        print(f"{stem}: loaded ({time.time() - t0:.0f} s)", flush=True)
    return libs, frag_init


def design_rows(units, protocols, average=False):
    """Design rows for (stem, protocol) units: intercept, protocol indicators (all but the first protocol),
    HEYA8 indicator, replicate indicators; with average, protocol columns set to their mean (consensus)."""
    stems = [f"{c}_rep{r}" for c, r in REPLICATES]
    cols = ["intercept"] + [f"protocol:{p}" for p in protocols[1:]] + ["cell:HEYA8"] + [f"rep:{s}" for s in stems]
    sd = np.r_[np.inf, np.full(len(protocols) - 1, np.inf), CELL_SD, np.full(len(stems), REP_SD)]
    X = np.zeros((len(units), len(cols)))
    for i, (stem, proto) in enumerate(units):
        X[i, 0] = 1.0
        for j, p in enumerate(protocols[1:]):
            X[i, 1 + j] = (1.0 / len(protocols)) if average else float(proto == p)
        X[i, len(protocols)] = float(stem.startswith("HEYA8"))
        X[i, len(protocols) + 1 + stems.index(stem)] = 1.0
    return X, sd


def fit(libs, keys, frag_init, tx_len, by_protocol, pool=None, protocols=PROTOCOLS, prev=None, verbose=True,
        groups=None, tau=TAU):
    """
    One fit over libraries `keys` ((stem, protocol)); units are stems, or (stem, protocol) when
    by_protocol. prev: an earlier fit's output, for warm-starting abundances and bias models.
    """
    units = sorted({(s, p) if by_protocol else s for s, p in keys})
    uidx = {u: i for i, u in enumerate(units)}
    pi = np.zeros((len(keys), len(units)))
    pi[np.arange(len(keys)), [uidx[k if by_protocol else k[0]] for k in keys]] = 1.0
    kw = {}
    tie = [[s for s, _ in units].index(s) for s, _ in units] if by_protocol else None
    if pool == "transcript":
        kw = dict(tie=tie, pool="transcript", pool_tau=tau)
    elif pool in ("gene_tied", "gene"):
        kw = dict(tie=tie, groups=groups)
        if pool == "gene":
            kw.update(pool="gene", pool_tau=tau)
    elif pool == "design":
        X, sd = design_rows(units, protocols)
        kw = dict(pool="design", design=X, design_sd=sd, pool_tau=TAU)
    em = JointEM([libs[k] for k in keys], pi=pi, **kw)
    models = copy.deepcopy(prev["models"]) if prev is not None else {}
    trunc, frag = {}, {}
    for i, (stem, proto) in enumerate(keys):
        if proto == "SR":
            if (stem, proto) not in models:
                fm = FragmentModel(tx_len)
                fm.pf = frag_init[stem].copy()
                models[(stem, proto)] = fm
            frag[i] = models[(stem, proto)]
        else:
            trunc[i] = models.setdefault((stem, proto), AnchoredTruncationModel(tx_len))
    init = None
    if prev is not None:
        pmap = {u: i for i, u in enumerate(prev["units"])}
        rows = [pmap.get(u, pmap.get(u[0]) if isinstance(u, tuple) else None) for u in units]
        if all(r is not None for r in rows):
            init = np.maximum(prev["res"].theta[rows], 1e-10)
    res = fit_bias(em, trunc=trunc, frag=frag, n_outer=8, run_kw=RUN_KW, verbose=verbose, init_theta=init,
                   outer_tol=1e-8 if prev is not None else None)
    return {"res": res, "units": units, "em": em, "models": models, "protocols": protocols}


def main(sgnex_dir, out_dir, only=None):
    sgnex_dir, out_dir = Path(sgnex_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, tx_len = read_fasta_lengths(sgnex_dir / "ref" / "hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa")
    t2g = pd.read_csv(sgnex_dir / "ref" / "t2g.tsv", sep="\t", header=None, index_col=0)[1]
    genes = pd.factorize(t2g.reindex(names))[0]
    libs, frag_init = load(sgnex_dir, names, tx_len)
    all_keys = sorted(libs)
    lr_keys = [k for k in all_keys if k[1] != "SR"]
    stems = [f"{c}_rep{r}" for c, r in REPLICATES]
    tpm = {s: {} for s in stems}
    summary, outs = [], {}
    plan = [("LR", lr_keys, False, None, PROTOCOLS[1:], None, TAU),
            ("joint", all_keys, False, None, PROTOCOLS, None, TAU),
            ("tpool", all_keys, True, "transcript", PROTOCOLS, "joint", TAU),
            ("design", all_keys, True, "design", PROTOCOLS, "tpool", TAU),
            ("LR_design", lr_keys, True, "design", PROTOCOLS[1:], "LR", TAU),
            ("gene", all_keys, True, "gene_tied", PROTOCOLS, "joint", TAU)]
    plan += [(f"gpool_t{t:g}", all_keys, True, "gene", PROTOCOLS, "joint", t) for t in (0.3, 1.0, 3.0)]
    if only:
        want = set(only.split(","))
        deps = {p[0]: p[5] for p in plan}
        for f in list(want):
            while deps.get(f):
                f = deps[f]
                want.add(f)
        plan = [p for p in plan if p[0] in want]
    for name, keys, by_proto, pool, protos, init, tau in plan:
        t0 = time.time()
        out = fit(libs, keys, frag_init, tx_len, by_proto, pool=pool, protocols=protos, prev=outs.get(init),
                  groups=genes, tau=tau)
        outs[name] = out
        res, units = out["res"], out["units"]
        for i, u in enumerate(units):
            stem, col = (u[0], f"{name}@{u[1]}") if by_proto else (u, name)
            tpm[stem][col] = molar_to_tpm(res.theta[i])
        if pool == "transcript":
            labels, prof = out["em"].pooled_profile()        # tie labels: index of each stem's first unit
            for j, lab in enumerate(labels):
                tpm[units[lab][0]][name] = molar_to_tpm(prof[j])
        if pool in ("gene_tied", "gene"):
            # consensus: geometric mean of the protocol units (shared within-gene fractions)
            for s in stems:
                rows = [i for i, u in enumerate(units) if u[0] == s]
                lg = np.log(np.maximum(res.theta[rows], 1e-300)).mean(axis=0)
                g = np.exp(lg - lg.max())
                tpm[s][name] = molar_to_tpm(g / g.sum())
        if pool == "design":
            cons = [(s, protos[0]) for s in stems]
            X, _ = design_rows(cons, protos, average=True)
            prof = out["em"].design_profile(X)
            for j, s in enumerate(stems):
                tpm[s][name] = molar_to_tpm(prof[j])
        summary.append({"fit": name, "units": len(units), "loglik": res.loglik[-1], "n_iter": res.n_iter,
                        "converged": res.converged, "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)
    for s in stems:
        df = pd.DataFrame(tpm[s], index=names)
        f = out_dir / f"{s}_tpm.tsv.gz"
        if only and f.exists():                    # add the new columns to an earlier run's output
            old = pd.read_csv(f, sep="\t", index_col=0)
            df = pd.concat([old.drop(columns=[c for c in df.columns if c in old.columns]), df], axis=1)
        df.to_csv(f, sep="\t", float_format="%.6g")
    fits_f = out_dir / "fits.tsv"
    summ = pd.DataFrame(summary)
    if only and fits_f.exists():
        summ = pd.concat([pd.read_csv(fits_f, sep="\t"), summ]).drop_duplicates("fit", keep="last")
    summ.to_csv(fits_f, sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
