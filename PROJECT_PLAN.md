# Project plan

Joint short-read (SR) + long-read (LR) isoform quantification across many samples: LR gives identifiability, SR gives coverage. Pure transcript-level quantification. Details and results are in [LABNOTEBOOK.md](LABNOTEBOOK.md).

Status as of 2026-11-03.

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

Open
- Evaluation design (not SR-only vs LR-only concordance): held-out LR likelihood on cells or donors, prediction of held-out LR from SR + remaining LR, agreement with the Parse Kinnex same-chemistry library.
- Sharing across cell types: tree or design-matrix prior (units are currently fitted independently).
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

1. Held-out evaluation for part 2.
2. Cross-cell-type prior (design matrix or tree) with variational inference, which also gives uncertainty.
3. Baselines on the bulk sets.
4. Atlas: ENCODE4 bulk LR + Tabula Sapiens SS2 with composition estimation.
