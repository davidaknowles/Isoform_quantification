# Project plan

Joint short-read (SR) + long-read (LR) isoform quantification across many samples: LR gives identifiability, SR gives coverage. Pure transcript-level quantification. Details and results are in [LABNOTEBOOK.md](LABNOTEBOOK.md).

Status as of 2026-11-24.

## 1. Bulk method development (WTC11, SG-NEx, simulations)

Done
- Joint SR+LR likelihood over shared molar abundances (`joli/joint.py`), SQUAREM, log-posterior stopping rule.
- Read-level inputs: salmon for SR, minimap2 with best-score ties for LR.
- Bias models: anchored censored-length read-end model for LR, LR length bias, jointly fitted SR fragment model (fragment-length distribution, coverage profile).
- Spike-in benchmarks: WTC11 SIRV-4 (read-end model and strict filter help every LR platform), SG-NEx SIRV/sequins (SR fragment model + all LR protocols best on sequins and SIRV E2).

Open
- Mixed results on SIRV-4 when SR is added. SR fragment G/C model implemented (optional): strong, reproducible G/C curve but little change in isoform fractions; salmon's full bias correction does no better. SR isoform error is fragment ambiguity, not bias.
- Baselines on SG-NEx spike-ins done (oarfish, lr-kallisto, NanoCount, IsoQuant, miniQuant; MPAQT failed): joint JOLI best on sequins, second to oarfish's truncation model on SIRVs; JOLI's LR read-end model loses to oarfish's truncation model. Next: improve the read-end model (compare with oarfish's), baselines on WTC11 SIRV-4.
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
- Same-cell SR + LR (IGVF mouse cortex B01_13H, Parse WT v2 + ONT, 5,687 nuclei, 13 cell types): short reads improve held-out ONT likelihood beyond the design-prior LR control at every LR depth (shared joint +0.10 nats per molecule at 5% LR training, +0.014 at 90%; within gene +0.03 / +0.002–0.006). First evidence that SR adds isoform information; unmatched donors (PBMC) never showed it. Next: SCOTCH human PBMC (Parse + ONT), CAST/EiJ nuclei with a strain-aware reference.
- design_gene (SR units = LR unit rescaled per gene; design prior on LR units): converges (SQUAREM, ~1 h at smoke scale, agrees with a 30k-step reference); best abundance on SG-NEx spike-ins; on held-out PBMC LR it loses like joint_gene (−0.049 at 1%), while the LR-only design control gains (+0.011).
- PBMC default = design_gene (SR informs gene and pre-mRNA totals; LR_design reported alongside); refitted on all data in 2 h 48 min (16 cores, 141 GB). Consensus isoform fractions are close to LR_design (median within-gene TV 0.015).
- Learn τ and the design sds (currently chosen by held-out likelihood); possibly separate τ for spliced and unspliced targets.
- Other held-out designs: whole LR libraries or donors, agreement with the Parse Kinnex same-chemistry library.
- Sharing across cell types: design-matrix prior implemented (intercept, platform, lineage, cell type); extend to donors and conditions.
- Posterior uncertainty, or identifiable isoform groups, for isoform-level scoring.
- Optional: 3' ends as poly(A)-site choice within an isoform.

## 3. Atlas (Tabula Sapiens/Muris SS2, Allen SMART-seq, ENCODE4 bulk LR)

Done
- Pipeline (`analyses/mouse_brain/fit.py`, reused for Tabula Sapiens): SS2 pseudobulks (region × class × age) + ENCODE bulk PacBio as read-share mixtures over region × class units; design_gene prior (intercept + region + lineage + class); SR weight 0.3 by default; never an SR-only isoform fit.
- Composition by gene-level deconvolution (`joli/deconv.py`: SS2 gene profiles from single-gene ECs, Allen MOp neurons added for cortex), fixed as read shares; estimated inside the EM it was not identified. Mouse brain: cortex 92% neuron, hippocampus 68%.
- Convergence: monotone MM with fixed composition (read-share mixtures, SQUAREM safeguard); design_gene M-step 5× faster.
- Mouse brain vs Joglekar 2024 single-cell LR truth: on genes with cell-type-specific isoform usage the joint consensus tracks Δψ (median r 0.67, sign agreement 0.81); absolute gene TV 0.15 vs 0.04 replicate noise (platform differences dominate).
- SR weight: strong shrinkage + low SR weight improves absolute TV but loses the truth's cell-type differences; SR carries that signal. Held-out bulk LR (thinned molecules, a whole region, or donors) does not select the weight the truth prefers (0.1–0.3).
- PBMC check (Ding Smart-seq2 + label-free bulk Kinnex predicting held-out cell-type Kinnex): no variant beats a pooled bulk LR profile; the labelled ceiling is only +0.0006 nats per molecule within gene.
- Tabula Sapiens muscle + heart: SS2 from the controlled on-cluster BAMs, 19 ENCODE PacBio libraries; fitted (muscle 77% fibres, heart 49% cardiomyocytes). Held-out heart donors: per-library composition gains overall (+0.011, partly leakage) but nothing within gene.

Open
- Bulk LR (PBMC Kinnex, heart) shows little cell-type isoform signal beyond the pooled tissue profile; only the mouse-brain truth confirms the joint model's cell-type isoform differences. Need more tissues with single-cell LR truth (human brain, other ScISOr-Seq/SnISOr-Seq sets).
- A data-driven criterion for the SR weight that sees cell types (held-out single-cell LR where available, or a combined SR + LR held-out likelihood).
- Leave-one-region-out for cortex; striatum and cerebellum predictions (no bulk LR).
- Allen human cortex (data use agreement unconfirmed); more Tabula Sapiens tissues.

## Engineering

Done
- Package `joli/`, uv venv, tests, LABNOTEBOOK.md.
- Rust kernels: E-step, expected-count scatter and weight gather for bias M-steps.
- Scaling: streaming library build, flat EC storage, EC ordering for memory locality, one-hot unit path, bias M-steps per unique (transcript, features) key.

Open
- Retire `AT_code/` and `JOLI_Kallisto/`.

## Next, in order

1. More single-cell LR truth for the atlas (human brain, other tissues); held-out single-cell LR as the SR-weight criterion (works on mouse brain).
2. Same-cell SR + LR: replicate the IGVF cortex gain (SCOTCH PBMC, CAST nuclei); variational inference for uncertainty.
3. Improve the LR read-end model against oarfish's truncation model; baselines on WTC11.
