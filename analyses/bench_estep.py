"""
Benchmark of the numpy and compiled (rust/joli_kernels) E-step on real libraries:
per E-step time by backend and thread count, and a full SR+LR fit on the simulation.

Run: python analyses/bench_estep.py
"""
import time, sys, numpy as np
sys.path.insert(0, "analyses")
from joli.joint import JointEM
from joli.salmon_io import library_from_salmon
from joli.paf import read_fasta_lengths, filter_alignments, library_from_alignments, read_paf
from sim_joint import SIM, quiet
from joli.load_tcc import load_tcc_data
names = quiet(load_tcc_data, SIM / "kallisto_output" / "ds_100_num1_aln_01_long").transcript_names
cases = {
  "sim SR (salmon)": [library_from_salmon("sr", "/gpfs/commons/home/daknowles/knowles_lab/daknowles/JOLI/salmon/quant/sim1_nobias", transcript_names=names)],
}
wnames, _ = read_fasta_lengths("/gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref/gencode.v38.transcripts.fa")
cases["WTC11 SR (salmon)"] = [library_from_salmon("sr", "/gpfs/commons/groups/knowles_lab/daknowles/JOLI/wtc11/salmon/ENCBS944CBA_bias", transcript_names=wnames)]
for name, libs in cases.items():
    fl_pos = sum(len(l.tcc.ec_counts) for l in libs)
    for backend, threads in [("numpy", None), ("rust", 1), ("rust", 2), ("rust", 4), ("rust", 8), ("rust", 16)]:
        em = JointEM(libs, backend=backend, threads=threads)
        theta = em.active / em.active.sum(1, keepdims=True)
        em._estep(theta)
        t = time.time(); n = 5
        for _ in range(n): em._estep(theta)
        dt = (time.time() - t) / n
        print(f"{name}: {em.libs[0].n_ec} ECs, {len(em.libs[0].tx)} positions | {backend:5s} threads={threads}: {dt*1000:.1f} ms/E-step", flush=True)
# full fit timing, sim SR+LR
lr = library_from_alignments("lr", filter_alignments(read_paf("/gpfs/commons/home/daknowles/knowles_lab/daknowles/JOLI/minimap2/sim1_long.paf.gz")), names)
for backend, threads in [("numpy", None), ("rust", 1), ("rust", 4), ("rust", 16)]:
    t = time.time(); r = JointEM(cases["sim SR (salmon)"] + [lr], units=[0, 0], backend=backend, threads=threads).run()
    print(f"sim SR+LR full fit {backend}: {time.time()-t:.1f} s, {r.n_iter} MM steps, loglik {r.loglik[-1]:.4f}", flush=True)
# many libraries (stand-in for pseudobulk fits): 16 copies of the sim SR library as separate units
libs16 = cases["sim SR (salmon)"] * 16
for backend, threads in [("numpy", None), ("rust", 1), ("rust", 4), ("rust", 16)]:
    em = JointEM(libs16, units=list(range(16)), backend=backend, threads=threads)
    theta = em.active / em.active.sum(1, keepdims=True)
    em._estep(theta)
    t = time.time(); n = 3
    for _ in range(n): em._estep(theta)
    print(f"16 libraries x {em.libs[0].n_ec} ECs | {backend:5s} threads={threads}: {(time.time() - t) / n * 1000:.1f} ms/E-step", flush=True)
