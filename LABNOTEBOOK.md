# Lab notebook

See `HANDOVER.md` for the code/data map left by the previous developer. This file records model components, findings and plans from the takeover onward.

## Current model (as inherited, 2026-10-27)

For sample $s \in \{1..S\}$ with transcript abundances $\theta_s \in \Delta^{T-1}$ ($T \approx 2\times10^5$ transcripts), reads are summarized as equivalence-class (EC) counts from `kallisto bus`/`bustools`. Each sample has its own EM. Samples are coupled by a shared prior $\theta_s \sim \mathrm{Dir}(\alpha)$, $\alpha \in \mathbb{R}_+^T$, with MAP M-step $\theta_{st} \propto n_{st} + \alpha_t$ where $n_{st}$ is the expected count of transcript $t$ in sample $s$.

- $\alpha$ is fit by Adam on the Dirichlet log-likelihood of the plug-in point estimates $\hat\theta_s$, with zeros floored to $10^{-10}$ (`JOLI_Kallisto/core/dirichlet_optimizer.py`). This is not a marginal likelihood, and it is the source of the $\alpha$ blow-ups (up to $1.9\times10^8$) and the `em_include_single_tx`/`gd_include_single_tx` workarounds.
- One concentration per transcript over the full transcriptome, so gene expression level and isoform usage share one prior and the mean and precision of each $\alpha_t$ are not separable.
- SR+LR combination: in `AT_code` (exp 2/5) SR and LR reads are pooled into one read dictionary with one $\theta$. `JOLI_Kallisto` has no joint SR+LR likelihood. SR and LR libraries can only enter as separate "samples" that share $\alpha$. Neither version models the different sampling of SR (fragments, proportional to $\theta_t \tilde{\ell}_t$ where $\tilde{\ell}_t$ is effective length) and LR (roughly one read per molecule, with length and 5'-truncation biases).
- The AT_code VI path (`--EM_type VI --dirichlet_process expectation_log_theta`) is the closest thing to a proper empirical Bayes fit of $\alpha$, since it uses $E_q[\log\theta]$.

## Findings

### 2026-10-27 Evaluation ground truth

The simulation (`sim_real_data`) provides three truths per sample: `sample{1,2}_gt.tsv` (generating abundances, ~150k expressed transcripts), `PB_sample*_gt.tsv` (realized LR read counts, ~71-79k transcripts) and `ill_sample*_gt.tsv` (RSEM estimate on simulated SR). `run_gt_comparison.py` scores against `PB_sample*_gt`, so a transcript that is expressed but drew no long reads counts as a false positive.

Across samples, the generating truth has Spearman 0.85 (union of nonzero), so the samples share substantial structure.

Re-scoring sim1 from the best inherited runs (SS = `exprmnt_2026_05_18__12_46_25`, MS = `exprmnt_2026_05_20__13_04_38`):

| | nonzero | FP vs PB GT | of those expressed in generating GT | FP TPM | Spearman vs generating GT | Pearson log1p vs generating GT |
|---|---|---|---|---|---|---|
| SS | 68,270 | 6,084 | 59% | 27,346 | 0.788 | 0.921 |
| MS | 85,351 | 16,066 | 72% | 9,197 | 0.830 | 0.936 |

Most MS "false positives" are expressed transcripts, and MS puts less total TPM on false positives than SS. Against the PB GT, MS is worse on GT-nonzero Spearman (0.814 vs 0.824) and Pearson (0.947 vs 0.967), so conclusions depend on which truth is used. Score against the generating truth from now on.

## Next steps (brainstorm, 2026-10-27)

Goal: joint SR+LR isoform quantification across many samples, such as single-cell metacells or fine cell states, where SR and LR libraries need not be paired one-to-one.

1. Reproduce and re-evaluate. Run the JK SS/MS pipeline end to end in a fresh env. Make the generating truth the default in `run_gt_comparison.py`. Add a per-gene isoform-fraction (DTU-style) metric, since multi-sample sharing should mostly help within-gene proportions.
2. Joint SR+LR likelihood. Keep one molar abundance $\theta_s$ per biological unit. Platform $p \in \{\mathrm{SR},\mathrm{LR}\}$ observes EC $e$ with probability $\propto \sum_t \theta_{st}\, b_{pt}\, c^{p}_{et}$, where $c^p_{et}$ is the platform's EC compatibility/likelihood and $b_{pt}$ is a platform sampling weight (effective length for SR, a learned length/truncation bias for LR). The E-steps stay per platform and the M-step pools expected molecule counts. MPAQT and miniQuant do this for bulk single samples. Our contribution is the multi-sample part.
3. Proper empirical Bayes for the prior. Replace plug-in Dirichlet ML with VI/CAVI ($q(\theta_s)$ Dirichlet, $\alpha$ updated from $E_q[\log\theta_s]$ by Minka fixed point or Adam on the ELBO). This should remove the single-tx flags and the $\alpha$ explosions, and it gives posterior uncertainty.
4. Structured prior (gene expression × within-gene fractions). Not pursued for now (decision 2026-10-27): stay with pure transcript-level quantification.
5. Flexible matching. Introduce latent units $k$ (cell states, metacells) with $\theta_k$. Each library $l$ (SR or LR, a cell, metacell or pseudobulk) is a mixture $\sum_k \pi_{lk}\theta_k$ with $\pi_{lk}$ known (same cells, shared barcodes), soft (cell-type composition) or estimated. Two ways to share across units:
   - Tree prior, $\theta_k \sim \mathrm{Dir}(\kappa\, \theta_{\mathrm{parent}(k)})$, so fine states with only SR borrow LR information from coarser ancestors.
   - Design matrix, $\theta_l = \mathrm{softmax}(\eta_l)$ with $\eta_l = B^\top x_l + \epsilon_l$, where $x_l \in \mathbb{R}^D$ codes the covariates of library $l$ (donor, tissue, cell type, condition, platform), $B \in \mathbb{R}^{D\times T}$ is shrunk (ridge or random effects), and $\epsilon_l \in \mathbb{R}^T$ is library-level noise. A tree is the special case where $x_l$ indicates the ancestors of $l$. This also allows crossed factors, e.g. SR from donors A,B and LR from donor C sharing a cell-type effect.
6. Single-cell specifics. Account for UMIs. 10x SR (3' or 5') is used only for barcodes, cell labels and expression, never for isoforms. Isoform-informative SR comes from full-length plate methods (Smart-seq2/3) or Parse/Split-seq (random-hexamer RT). Use lr-kallisto/bustools barcode handling. Batch the EM on GPU (torch sparse) across thousands of libraries.
7. Simulation. Extend `Simulations/` to generate cell states with planted DTU, and SR/LR designs with mismatched resolution (e.g. SR per fine state, LR per coarse type).
8. Baselines. lr-kallisto, oarfish, IsoQuant, Bambu/Bambu-Clump, MPAQT, miniQuant, Isosceles (single-cell pseudobulk).
9. Housekeeping. Move to a uv-managed venv, turn `JOLI_Kallisto/core` into an importable package (replacing the `sys.path` inserts), and retire `AT_code` once JK covers VI. The target is a new paper, not a revision of the 2025 JOLI preprint.

## Data survey (2026-10-27)

Matching types: exact (same library cDNA or same cells), same biosample, same donor, or only same cell type/tissue (different donors).

### On the cluster

| Dataset | SR | LR | Matching | Notes |
|---|---|---|---|---|
| LRGASP-style WTC11/H1 (student's ENCODE 2026 download, ~52 GB) | Illumina, 9 experiments | PacBio/ONT cDNA, dRNA, 6 experiments | same cell line, replicates | bulk, best for method development; also the older WTC-11 PacBio+Illumina and H1 Illumina/ONT/PacBio alignments |
| ENCODE4 PacBio human (Megan, `encode_pacbio`) | not downloaded | 138 libraries, 95 experiments, 51 tissue, 61 cell line, 22 in vitro differentiated | ENCODE often has SR on the same biosample (to check via the ENCODE API) | DLPFC ×9, heart chambers, lung, colon, adrenal, ovary, HL-60 ×8 |
| ENCODE4 PacBio mouse (Megan, `2025_mouse_longread`) | to check | yes | to check | mouse tissues |
| ENCODE4 K562/HepG2 LR (Julia, `ENCODE4_LRS_LRP2_demo_data`) | – | yes, aligned | – | with mass spec |
| PacBio Kinnex PBMC (`sc/pacbio_kinnex_pbmc`, 1.4 TB) | 10x barcodes only | 7 × 10x 3'/5' (Revio, Vega) + 1 × Parse WT | same cells LR↔barcodes; donors differ from the Parse SR sets | the Parse Kinnex library has the same chemistry as Parse SR |
| Parse PBMC whole cells (tealeaf: GSE327907, GSE334840, GSE320507, GSE285843, GSE246624, GSE295527) | Parse WT, isoform-informative | – | cell type only (to Kinnex) | 2–46 donors per series, stimulation and long-COVID designs |
| Parse brain nuclei (tealeaf: GSE233208 human, GSE189033 mouse) | Parse WT | – | – | nuclei, so intronic/precursor dominated |
| Tabula Muris (`data/archive/tabula_muris/smart_seq`, ~7.4k files) | Smart-seq2 FACS | – | tissue/cell type to mouse LR | Tabula Muris Senis has metadata only on disk; reads on AWS |
| Tabula Sapiens | Smart-seq2 (`TabulaSapiens_ss2only.h5ad`, Leaflet/SpliZ outputs; raw BAMs on AWS) | – | tissue/cell type to ENCODE4 human tissue LR | the lab's IsoformGazer uses TS v2 SS2 |
| Allen human cortex (BRAIN/NeMO, controlled, DAC-approved for DAK and Karin) | SMART-seq v4 nuclei, ~50k nuclei, 6 cortical regions (MTG, M1C, CgGr, V1C, S1C, A1C), raw FASTQ (6.7 TB) + 208 MTG neurosurgical (38 GB) | – | cell type/region to ENCODE4 DLPFC PacBio (×9) | ~90% neurons (NeuN sorted), with Allen taxonomy labels; GRU, MTG limited to HMB |
| Allen mouse MOp (Karin, processed) | SMART-seq v4 cells and nuclei, gene counts only on disk | – | cell type to mouse brain LR | raw reads are open on NeMO; the larger Allen mouse cortex+hippocampus SMART-seq set is also open |
| GTEx ONT (`ont_gtex`) | GTEx Illumina (dbGaP) | junction BEDs only on disk | same sample | full data: Glinos et al. 2022, 88 samples, 15 tissues |

### Public, not yet downloaded

- SG-NEx (Chen et al. 2025 Nat Methods, AWS open data): 7+ cell lines, Illumina + ONT dRNA/cDNA/PCR-cDNA + PacBio, replicates, spike-ins. Best bulk multi-platform benchmark with known spike-in truth.
- LR-Split-seq (Rebboah et al. 2021 Genome Biol): Split-seq SR and LR on the same cells (C2C12). The Mortazavi lab IGVF mouse founder-strain data may extend this to tissues (to check).
- 10x + LR same-cell sets (ScISOr-Seq/SnISOr-Seq brain, scNanoGPS, SiCeLoRe, FLT-seq, MAS-seq): useful only as LR-per-cell with barcodes.

NIMH_LRS (PacBio HiFi, ~99 samples) is controlled access and not used.

## Paper plan (2026-10-27)

A new paper in three parts.

1. Method development on bulk SR+LR. WTC11/H1 replicates (LRGASP/ENCODE) and SG-NEx (spike-ins, several platforms), plus simulations with known truth. Establishes the joint SR+LR likelihood and the multi-sample empirical Bayes prior against lr-kallisto, oarfish, IsoQuant, MPAQT and miniQuant.
2. Single cell/nucleus with Parse SR and LR. Parse PBMC SR (many donors and conditions) with Kinnex PBMC LR (different donors, cells barcoded so LR is per cell). Matching by cell type through the design matrix or tree. The Parse Kinnex library is a same-chemistry anchor. Held-out LR cells or donors give a real-data check of SR-only predictions.
3. Atlas demo: the best available cell-type-resolved map of human and mouse isoform usage. Isoform-informative SR from Tabula Sapiens SS2, Allen human cortex SMART-seq v4 and Parse brain nuclei (human); Tabula Muris/Senis SS2 and Allen mouse SMART-seq (mouse). LR from ENCODE4 PacBio tissues (human and mouse, bulk) and single-cell LR where available. Bulk LR libraries enter as mixtures over cell types with composition $\pi_{lk}$ estimated from single-cell references, and a tissue × cell-type design matrix (or tree) shares information across tissues, regions and studies.

Brain is the tissue where all pieces overlap for human: Allen SMART-seq v4 (cell types × 6 regions), Parse brain nuclei (GSE233208), and ENCODE4 DLPFC PacBio ×9.
