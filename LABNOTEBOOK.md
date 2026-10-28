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

Within-gene isoform fractions ($\psi_t = \mathrm{TPM}_t / \sum_{t' \in g(t)} \mathrm{TPM}_{t'}$, genes with ≥2 transcripts and true gene TPM ≥1, ~12.7k genes) also favour MS (`analyses/sim_rescore.py`, `joli/evaluate.py`): sim1 $\psi$ Pearson 0.891 → 0.905, median per-gene total-variation error 0.162 → 0.140, major-isoform accuracy 0.757 → 0.777. JK SS reproduces lr-kallisto exactly.

### 2026-10-27 Packaging

`JOLI_Kallisto/core` is now the package `joli/` (relative imports, `pyproject.toml`, installed editable into a uv venv with CPU torch). The student's scripts and tests import `joli.*`. Two tests already failed on the original code (`test_JolitoKallisto.py` looks for `main_joli.py` in `test/`; `test_main_multisample_joli.py` calls an old signature); one stale attribute in `test_em_algorithm.py` was fixed. New tests are in `tests/` (pytest).

### 2026-10-27 Joint SR+LR likelihood (`joli/joint.py`)

Units $k$ have molar abundances $\theta_k \in \Delta^{T-1}$. Library $l$ observes $\mu_l = \sum_k \pi_{lk}\theta_k$, with $\pi \in \mathbb{R}_+^{L\times K}$ the molecule fractions of each unit in each library (one-hot for a library from one sample). A read of library $l$ falls in EC $e$ with probability $P(e \mid l) = \sum_{t\in e} \mu_{lt} c_{let} / \sum_t \mu_{lt} b_{lt}$, where $b_{lt}$ is the sampling rate of transcript $t$ (effective length for SR, 1 for LR) and $c_{let}$ the relative compatibility weight (1 for now; bias models enter through $b$ and $c$). With $b$ the effective length and $c=1$ this is kallisto's SR likelihood in molar units. The $-N_l \log B_l$ normaliser, $B_l = \sum_t \mu_{lt} b_{lt}$, is minorised by its tangent, giving the MM update $\theta_{kt} = (Z_{kt} + a_{kt})/(D_{kt} + \lambda_k)$ with $Z_{kt}$ the expected reads of $(k,t)$ summed over libraries, $D_{kt} = \sum_l N_l \pi_{lk} b_{lt}/B_l$, Dirichlet pseudo-counts $a_{kt}$, and $\lambda_k$ normalising. For one LR library this is JoliEM's MAP update. Tests check agreement with JoliEM (LR) and kallisto's $\rho$ (SR), monotone likelihood, recovery of mixture components, and that joint beats either platform on synthetic data.

Iterates use SQUAREM (scheme S3) with a fallback to the plain MM step when the extrapolated point has lower log posterior. On the 0.9M-read LR sample it reaches the plain-EM optimum (log-likelihood −8,018,894.43) in 372 MM steps (6 s) against ~2,000 plain steps. After convergence, transcripts with < 0.01 expected reads are zeroed and one more MM step is run (kallisto's finalRound): many MLEs sit on the simplex boundary, and without this the number of nonzero transcripts depends on the iteration count (73k at 500 plain steps, 66k at 2,000, 81k with SQUAREM). Spearman on the union of nonzero transcripts tracks this count (0.808 → 0.794 between 500 and 2,000 steps at near-identical likelihood), so it is a poor primary metric. Use log-Pearson, MARD and the $\psi$ metrics.

lr-kallisto's long-read EM (single-transcript ECs left out of the EM loop and added afterwards) is not the MLE. On sim1 LR its log-likelihood is −8,039,332 against −8,018,894 for the MLE, with lower log-Pearson to the generating truth (0.921 vs 0.925).

Simulation, SR+LR per sample against the generating truth (`analyses/sim_joint.py`, 9.5M SR pairs and 0.9M LR reads per sample, no cross-sample prior):

| sample | fit | log-Pearson | MARD | FN | FP | $\psi$ Pearson | gene TV median | major-isoform acc |
|---|---|---|---|---|---|---|---|---|
| sim1 | LR | 0.925 | 0.674 | 88,456 | 2,196 | 0.895 | 0.141 | 0.772 |
| sim1 | SR | 0.893 | 0.662 | 74,387 | 5,031 | 0.830 | 0.200 | 0.687 |
| sim1 | SR+LR | 0.933 | 0.608 | 71,442 | 4,156 | 0.899 | 0.125 | 0.780 |
| sim2 | LR | 0.921 | 0.642 | 83,525 | 2,214 | 0.893 | 0.143 | 0.765 |
| sim2 | SR | 0.885 | 0.642 | 71,635 | 4,802 | 0.828 | 0.209 | 0.678 |
| sim2 | SR+LR | 0.929 | 0.581 | 68,109 | 4,028 | 0.897 | 0.127 | 0.770 |

SR+LR improves on both platforms on every metric except FP count (between the two). SR alone is worse than LR for isoform fractions despite 10× the reads. Fits take 5 s (LR), 60–100 s (SR) and 11–23 s (SR+LR, LR resolves the ambiguity SR converges slowly on) in numpy.

### 2026-10-28 Read-level inputs: salmon for SR, minimap2 for LR

Bias models need per-read positions, which kallisto's TCCs drop. Two loaders now produce JointEM libraries from aligners:

- `joli/salmon_io.py` reads salmon's rich equivalence classes (`salmon quant --dumpEq --dumpEqWeights`) and its effective lengths (which absorb `--seqBias/--gcBias/--posBias`). The dumped weights are salmon's combined weights $v_{et} \propto w_{et}/l_t$ (auxiliary weight over effective length), normalised within the class. With $b_t = l_t$ and $c_{et} = v_{et} l_t$, salmon's own estimate is a fixed point of JointEM (expected reads agree to 80 of 9.8M); with $c_{et}=v_{et}$ the estimates disagree badly.
- `joli/paf.py` reads minimap2 transcriptome alignments (`-c -N 100`), keeps per read the alignments within a score threshold of the best, and collapses reads into ECs, optionally keyed by integer per-alignment features (exact for any bias model that depends on those features only).

The score threshold matters a lot. On the simulation the 0.95-of-best threshold (oarfish's default) admits isoforms differing by whole exons (median score gap among admitted non-best alignments 122) and gives LR-only log-Pearson 0.74. Keeping only best-score ties gives 0.96, well above lr-kallisto's TCCs (0.925). Soft weights $\exp(-\beta\,\Delta\mathrm{AS})$ collapsed into rich ECs do not recover this (0.83 even at $\beta=4$): averaging within a transcript set mixes tied reads with reads that have large gaps, the problem salmon's range factorisation addresses.

### 2026-10-28 Long-read truncation model (`joli/lr_bias.py`)

A read aligned to transcript $t$ of length $L_t$ at $[s,e)$ leaves $d_5 = s$ and $d_3 = L_t - e$ uncovered bases. $P(d_5, d_3 \mid t) = q_5(d_5 \mid L_t)\, q_3(d_3 \mid L_t)$ with $q(d\mid L) = (p_j/w_j)/F(L)$ for $d$ in bin $j$ of width $w_j$, $p \in \Delta^{J-1}$ over $J=19$ log-spaced bins, and $F(L) = \sum_j p_j o_j(L)$ the probability mass that fits in a transcript of length $L$ ($o_j(L)$ the fraction of bin $j$ below $L+1$). Weights enter JointEM as $c_{et} = q_5 q_3$. Given expected reads per alignment, the M-step maximises $\sum_j A_j \log p_j - \sum_t Z_t \log F_t(p)$ ($A_j$ expected reads in bin $j$, $Z_t$ expected reads of $t$), solved by the fixed point $p_j \propto A_j / \sum_t Z_t o_{tj}/F_t$ restricted to bins reachable by some expressed transcript. Outer iterations alternate this with warm-started JointEM runs. The model distinguishes a short isoform from the 3' end of a long one, which plain LR EM cannot. On a toy simulation of nested isoforms it cuts the L1 error from 1.03 to 0.08.

Simulation, LR input (`analyses/sim_lr_paf.py`, sim1; sim2 within 0.01):

| LR input | fit | log-Pearson | MARD | $\psi$ Pearson | gene TV median | major-isoform acc |
|---|---|---|---|---|---|---|
| lr-kallisto TCC | LR | 0.925 | 0.674 | 0.895 | 0.141 | 0.772 |
| minimap2, AS ≥ 0.95 best | LR | 0.740 | 0.752 | 0.721 | 0.327 | 0.567 |
| minimap2, best ties | LR | 0.961 | 0.582 | 0.935 | 0.053 | 0.865 |
| minimap2, best ties + truncation | LR | 0.968 | 0.573 | 0.942 | 0.041 | 0.872 |
| lr-kallisto TCC | SR+LR | 0.934 | 0.605 | 0.900 | 0.123 | 0.781 |
| minimap2, best ties | SR+LR | 0.955 | 0.547 | 0.922 | 0.082 | 0.819 |
| minimap2, best ties + truncation | SR+LR | 0.961 | 0.541 | 0.927 | 0.075 | 0.824 |

The learned profiles are 3'-anchored (87% of reads within 1 base of the 3' end) with 37% full length at the 5' end and the rest spread over 3–100 bases.

With the better LR input, adding SR now makes $\psi$ worse (gene TV 0.053 → 0.082), in every LR-depth stratum including genes with > 1,000 LR reads (0.001 → 0.004), so the SR contribution is biased rather than noisy. The SR generation itself matches the model: true SR read counts per transcript (recovered from RSEM read names) match truth × effective length within Poisson noise, start positions are uniform, and SR alone has the expected log-likelihood gap between MLE and truth (52k for ~81k nonzero transcripts). Gene-level SR and LR sampling rates agree with the truth within 3% across length deciles, and changing $b$ for the 100-nt poly(A) tail that the simulator's RSEM reference appends had no effect. The remaining suspect is read-to-EC assignment (true transcript missing from the EC), not yet tested. The sim's LR reads may also be cleaner than real ones (best-tie assignment is nearly an oracle).

Short-read input (`analyses/sim_salmon.py`): salmon selective alignment is slightly better than kallisto (SR-only log-Pearson 0.898 vs 0.893, $\psi$ 0.835 vs 0.830; salmon's own quant.sf 0.901). Rich weights make no difference here. Salmon's bias correction slightly hurts (0.889), as expected for data simulated without bias.

### 2026-10-28 WTC11, four platforms on three biosamples

ENCSR450EXF (Illumina SE100), ENCSR507JOF (PacBio poly(A)), ENCSR309IKK (PacBio CapTrap) and ENCSR392BGY (ONT direct RNA) share biosamples ENCBS944CBA, ENCBS593PKA and ENCBS474NOC. Reference GENCODE v38. SR with salmon (genome decoys), LR with minimap2 (`analyses/wtc11/align.sh`, `quant.py`, `compare.py`).

Long-read platforms agree. PB vs ONT gives $\psi$ Pearson 0.94 and gene TV 0.08 (genes with ≥ 2 isoforms, ≥ 5 TPM in both). SR agrees with none of them ($\psi$ Pearson ≈ 0.59, gene TV ≈ 0.58) but is reproducible across biosamples ($\psi$ Pearson 0.97). The SR libraries are total RNA-seq (rRNA-depleted). 55–60% of reads map to transcripts rather than genome decoys, and 73% of SR TPM goes to misc_RNA (7SK/7SL), snRNA, ribozymes and snoRNAs, against 0% in PB. Retained-intron transcripts are 10% of the protein-coding plus retained-intron TPM in SR, against 2% in PB. Every WTC11 and H1 SR experiment in the student's ENCODE set is total RNA (the others are 32-nt PE), so WTC11 has no poly(A) SR partner for its LR. salmon's sequence and positional bias correction lowers SR–LR agreement further ($\psi$ Pearson 0.44).

The truncation model slightly lowers cross-platform LR agreement (PB vs ONT $\psi$ Pearson 0.941 → 0.920). Agreement is not accuracy, since plain models on two platforms can share the same error (e.g. assigning truncated reads to short isoforms). Real-data accuracy needs spike-ins with known isoform molarities: SIRVs in the WTC11 PacBio libraries, and SIRV/sequins in SG-NEx.

## Requirements added 2026-10-27

- Primer and coverage bias awareness. Parse (oligo-dT and random-hexamer RT in the same cell), Smart-seq2 (3' bias in degraded samples) and LR (5' truncation, length-dependent capture) all violate $c_{let}=1$, $b_{lt}=$ effective length. The bias model enters through $b_{lt}$ (expected reads per molecule, integrating a positional density over transcript coordinates) and $c_{let}$ (the share of that density falling in positions that produce EC $e$). This needs read positions, which kallisto's TCC output drops. Options are tealeaf's aligner (it already computes primer-specific read opportunities for Parse), salmon/alevin-fry selective alignment, or minimap2 for LR. Bias parameters (a few per library and primer, e.g. a positional density on relative coordinates) can be fitted by an outer M-step.
- Atlas efficiency. Thousands of libraries × ~200k transcripts. The E-step is a sparse gather/scatter over (EC, transcript) positions and parallelises over libraries. Plan is a Rust kernel with a C ABI called through ctypes (the pattern of `tealeaf/rust/native_align`), with numpy kept as the reference implementation, plus SQUAREM.
- Cell-type composition for the atlas. Bulk tissue LR libraries are mixtures over cell types ($\pi_{lk}$ above). $\pi_{lk}$ should be estimated, not fixed: initialise from single-cell references (deconvolution on gene-level counts) and update within the EM. The M-step for $\pi_l$ has the same MM form, with a Dirichlet prior centred on the reference-based estimate. Composition is in molecule units, so it has to account for cell-type differences in mRNA content.

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
