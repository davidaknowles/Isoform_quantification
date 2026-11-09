"""Simulated cell types x platforms for design-prior convergence checks (used by tests and benchmarks)."""
import numpy as np

from joli.joint import Library
from joli.load_tcc import FlatECs, TCCData


def design_problem(rng, K=6, G=300, iso=3, dev=0.4, plat_dev=0.5, lr_reads=3000, sr_reads=200000):
    """
    K cell types with log abundances = shared base + cell-type deviation (sd dev) + platform effect
    (sd plat_dev, shared across cell types); per cell type an SR library (80% gene-level ECs) and an
    LR library (transcript ECs). Returns libraries, units (K x 2), design X, truth (2K, T), gene.
    """
    T = G * iso
    names = [f"t{i}" for i in range(T)]
    gene = np.repeat(np.arange(G), iso)
    base = rng.normal(0, 1.5, T)
    plat = rng.normal(0, plat_dev, T)
    libs, X, truth = [], [], []
    for k in range(K):
        ct = base + rng.normal(0, dev, T)
        for p, n in [(0, sr_reads), (1, lr_reads)]:
            x = ct + p * plat
            th = np.exp(x) / np.exp(x).sum()
            truth.append(th)
            if p == 0:
                ecs = [np.arange(g * iso, (g + 1) * iso) for g in range(G)] + [np.array([t]) for t in range(T)]
                probs = np.r_[0.8 * np.bincount(gene, th), 0.2 * th]
            else:
                ecs, probs = [np.array([t]) for t in range(T)], th
            counts = rng.multinomial(n, probs / probs.sum()).astype(float)
            ptr = np.r_[0, np.cumsum([len(e) for e in ecs])]
            tcc = TCCData(ec_counts=counts, ec_transcripts=FlatECs(ptr, np.concatenate(ecs).astype(np.int32)),
                          transcript_names=names, total_reads=n, n_targets=T, sample_dir="")
            libs.append(Library(f"k{k}p{p}", tcc, b=np.ones(T)))
            X.append(np.r_[1.0, p, np.eye(K)[k]])
    return libs, np.array(X), np.array(truth), gene
