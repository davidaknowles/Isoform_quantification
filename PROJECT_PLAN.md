# Project plan

Joint short-read (SR) + long-read (LR) isoform quantification across many samples: LR gives identifiability, SR gives coverage. Pure transcript-level quantification. Details and results are in [LABNOTEBOOK.md](LABNOTEBOOK.md).

Status as of 2026-11-10.

## 1. Bulk method development (WTC11, SG-NEx, simulations)

Done
- Joint SR+LR likelihood over shared molar abundances (`joli/joint.py`), SQUAREM, log-posterior stopping rule.
- Read-level inputs: salmon for SR, minimap2 with best-score ties for LR.
- Bias models: anchored censored-length read-end model for LR, LR length bias, jointly fitted SR fragment model (fragment-length distribution, coverage profile).
- Spike-in benchmarks: WTC11 SIRV-4 (read-end model and strict filter help every LR platform), SG-NEx SIRV/sequins (SR fragment model + all LR protocols best on sequins and SIRV E2).

Open
- Mixed results on SIRV-4 when SR is added; abundance correlation drops in joint fits (SR fragment model needs sequence/GC bias).
- Baselines: lr-kallisto, oarfish, IsoQuant, MPAQT, miniQuant.
- Empirical Bayes prior across samples (variational inference in place of plug-in Dirichlet).

## 2. Single cell: Parse SR + 10x Kinnex LR (PBMC, different donors)

Done
- Pipeline: alevin-fry read positions, CellTypist labels, libraries per cell type × kit × source library.
- Primer-specific read-position models; poly(dT) internal-priming scale on unspliced targets.
- Joint fit on all ~1B molecules (671M EC positions), all stages converge; unspliced share 6–27% per cell type.
- Findings: Parse coverage is broad for both primers; 10x Kinnex reads often end upstream of GENCODE 3' ends.
- Default joint fit = design prior across cell types and platforms (intercept + platform + lineage + cell type effects, unit noise); previous default (transcript-level platform pooling) refitted on all data in 3 h 15 min; full refit with the design prior pending faster convergence.
- Held-out LR evaluation (fixed 10% of LR molecules; training on 1–90%). With eps-smoothing, platform pooling and the design prior appeared to gain a lot (+0.14 and +0.28 nats per molecule at 1%), but rescoring only molecules with positive probability under every fit shows those gains came from the ~2% of held-out molecules that LR alone leaves at zero probability. On the common molecules platform pooling loses to LR alone and only the design prior gains, modestly (+0.010 per molecule at 1%, +0.002 at 20%). An LR-only fit with the same design prior gains as much (+0.011 at 1%), so the gain is cross-cell-type sharing; Parse SR (other donors) adds nothing to held-out 10x LR prediction.

Open
- Evaluate what SR adds with truth not defined by one platform: spike-ins (SG-NEx, WTC11 SIRVs) for the design/pooling models, or same-cell SR and LR (e.g. LR-Split-seq, IGVF Parse SR + ONT on the same nuclei); held-out LR likelihood cannot credit SR for correcting LR biases.
- Convergence of the design fit (~10k MM steps): bias models fixed from the warm start, Anderson overhead, step-count benchmarks on one node.
- Learn τ and the design sds (currently chosen by held-out likelihood); possibly separate τ for spliced and unspliced targets.
- Other held-out designs: whole LR libraries or donors, agreement with the Parse Kinnex same-chemistry library.
- Sharing across cell types: design-matrix prior implemented (intercept, platform, lineage, cell type); extend to donors and conditions.
- Posterior uncertainty, or identifiable isoform groups, for isoform-level scoring.
- Optional: 3' ends as poly(A)-site choice within an isoform.

## 3. Atlas (Tabula Sapiens/Muris SS2, Allen SMART-seq, ENCODE4 bulk LR)

Done
- Data survey (LABNOTEBOOK.md).

Open
- Everything: bulk LR libraries as mixtures over cell types with composition estimated in the EM, tissue × cell-type sharing, brain as the overlap tissue (Allen human cortex data use agreement unconfirmed).

## Engineering

Done
- Package `joli/`, uv venv, tests, LABNOTEBOOK.md.
- Rust kernels: E-step, expected-count scatter and weight gather for bias M-steps.
- Scaling: streaming library build, flat EC storage, EC ordering for memory locality, one-hot unit path, bias M-steps per unique (transcript, features) key.

Open
- Retire `AT_code/` and `JOLI_Kallisto/`.

## Next, in order

1. Design-fit convergence; evaluation of what SR adds with platform-independent truth (spike-ins, same-cell SR + LR); then refit all data; variational inference for uncertainty.
2. Baselines on the bulk sets.
3. Atlas: ENCODE4 bulk LR + Tabula Sapiens SS2 with composition estimation.
