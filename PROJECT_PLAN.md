# Project plan

Joint short-read (SR) + long-read (LR) isoform quantification across many samples: LR gives identifiability, SR gives coverage. Pure transcript-level quantification. Details and results are in [LABNOTEBOOK.md](LABNOTEBOOK.md).

Status as of 2026-11-18.

## 1. Bulk method development (WTC11, SG-NEx, simulations)

Done
- Joint SR+LR likelihood over shared molar abundances (`joli/joint.py`), SQUAREM, log-posterior stopping rule.
- Read-level inputs: salmon for SR, minimap2 with best-score ties for LR.
- Bias models: anchored censored-length read-end model for LR, LR length bias, jointly fitted SR fragment model (fragment-length distribution, coverage profile).
- Spike-in benchmarks: WTC11 SIRV-4 (read-end model and strict filter help every LR platform), SG-NEx SIRV/sequins (SR fragment model + all LR protocols best on sequins and SIRV E2).

Open
- Mixed results on SIRV-4 when SR is added. SR fragment G/C model implemented (optional): strong, reproducible G/C curve but little change in isoform fractions; salmon's full bias correction does no better. SR isoform error is fragment ambiguity, not bias.
- Baselines: lr-kallisto, oarfish, IsoQuant, MPAQT, miniQuant.
- Empirical Bayes prior across samples (variational inference in place of plug-in Dirichlet).

## 2. Single cell: Parse SR + 10x Kinnex LR (PBMC, different donors)

Done
- Pipeline: alevin-fry read positions, CellTypist labels, libraries per cell type × kit × source library.
- Primer-specific read-position models; poly(dT) internal-priming scale on unspliced targets.
- Joint fit on all ~1B molecules (671M EC positions), all stages converge; unspliced share 6–27% per cell type.
- Findings: Parse coverage is broad for both primers; 10x Kinnex reads often end upstream of GENCODE 3' ends.
- Default joint fit = design_gene: design prior across cell types (intercept + lineage + cell type effects, unit noise) on the LR units, SR units = LR units rescaled per gene.
- Held-out LR evaluation (fixed 10% of LR molecules; training on 1–90%). With eps-smoothing, platform pooling and the design prior appeared to gain a lot (+0.14 and +0.28 nats per molecule at 1%), but rescoring only molecules with positive probability under every fit shows those gains came from the ~2% of held-out molecules that LR alone leaves at zero probability. On the common molecules platform pooling loses to LR alone and only the design prior gains, modestly (+0.010 per molecule at 1%, +0.002 at 20%). An LR-only fit with the same design prior gains as much (+0.011 at 1%), so the gain is cross-cell-type sharing; Parse SR (other donors) adds nothing to held-out 10x LR prediction.

Open
- SG-NEx spike-ins (done): only the shared fit lets SR improve isoform fractions (SIRV-4 gene TV 0.230 → 0.143); SR's isoform information comes mostly through shared gene totals (length-weighted abundance of within-gene-ambiguous fragments), which decoupled models discard. Default model by design: shared fit for matched samples, design prior for unmatched donors.
- Same-cell or same-donor SR + LR (e.g. LR-Split-seq, IGVF Parse SR + ONT on the same nuclei) to test SR's isoform contribution in single cell, where gene totals can be shared.
- design_gene (SR units = LR unit rescaled per gene; design prior on LR units): converges (SQUAREM, ~1 h at smoke scale, agrees with a 30k-step reference); best abundance on SG-NEx spike-ins; on held-out PBMC LR it loses like joint_gene (−0.049 at 1%), while the LR-only design control gains (+0.011).
- PBMC default = design_gene (SR informs gene and pre-mRNA totals; LR_design reported alongside); refitted on all data in 2 h 48 min (16 cores, 141 GB). Consensus isoform fractions are close to LR_design (median within-gene TV 0.015).
- Learn τ and the design sds (currently chosen by held-out likelihood); possibly separate τ for spliced and unspliced targets.
- Other held-out designs: whole LR libraries or donors, agreement with the Parse Kinnex same-chemistry library.
- Sharing across cell types: design-matrix prior implemented (intercept, platform, lineage, cell type); extend to donors and conditions.
- Posterior uncertainty, or identifiable isoform groups, for isoform-level scoring.
- Optional: 3' ends as poly(A)-site choice within an isoform.

## 3. Atlas (Tabula Sapiens/Muris SS2, Allen SMART-seq, ENCODE4 bulk LR)

Done
- Data survey (LABNOTEBOOK.md).
- Mouse-brain pilot: Tabula Muris Senis Smart-seq2 (46 region × class × age groups) + ENCODE bulk PacBio (cortex, hippocampus) as mixtures over region × class units with composition estimated, design_gene prior; scored against Joglekar 2024 P56 ScISOr-Seq2. On genes with cell-type-specific isoform usage in the truth, the joint consensus tracks Δψ (median r 0.67).
- Mouse brain with PBMC-style shrinkage + SR tempering: absolute TV improves (0.160 → 0.125) but DIU against Joglekar drops (Δψ r 0.63 → 0.49); SR carries the cell-type isoform signal. Held-out bulk LR (thinned molecules, or a whole region left out) does not select the weight the truth prefers; the truth points to SR weight ~0.1–0.3 under the default prior.
- PBMC atlas check (Smart-seq2 of other donors + label-free bulk Kinnex, deconvolved composition, predicting held-out cell-type Kinnex): no variant beats a pooled bulk LR profile (best −0.06 vs −0.02 nats per molecule overall, −0.004 vs −0.001 within gene; labelled ceiling +0.075 / +0.0006). Needs strong class shrinkage (class/lineage sd ≤ 0.1) and a tempered SR likelihood (w 0.02–0.1); these Kinnex data have little cell-type isoform signal.
- Composition: estimated in the EM it was not identified (cortex 71% OPC; those units became bulk-like). Now fixed by gene-level deconvolution (Allen MOp neurons added to the cortex reference; cortex 73% neuron, hippocampus 61%) as read shares (`JointEM(mixture="reads")`), which keeps design_gene MM monotone; converged in 2.5 h; no joint unit fails against the truth any more.

Open
- Platform/protocol differences dominate absolute isoform-fraction error (every estimate 0.10–0.18 median gene TV from the truth vs 0.04 replicate noise).
- Tabula Sapiens pilot (muscle TSP4, heart TSP12 Smart-seq2; ENCODE4 psoas and heart PacBio, 19 libraries aligned): groups built, SR reads blocked (S3 objects need credentials; lab access via Karin). Allen human cortex (data use agreement unconfirmed).

## Engineering

Done
- Package `joli/`, uv venv, tests, LABNOTEBOOK.md.
- Rust kernels: E-step, expected-count scatter and weight gather for bias M-steps.
- Scaling: streaming library build, flat EC storage, EC ordering for memory locality, one-hot unit path, bias M-steps per unique (transcript, features) key.

Open
- Retire `AT_code/` and `JOLI_Kallisto/`.

## Next, in order

1. Same-sample SR + LR single-cell data to test whether SR helps isoforms when gene totals can be shared; variational inference for uncertainty.
2. Baselines on the bulk sets.
3. Atlas: ENCODE4 bulk LR + Tabula Sapiens SS2 with composition estimation.
