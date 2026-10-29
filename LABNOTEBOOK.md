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

With the better LR input, adding SR now makes $\psi$ worse (gene TV 0.053 → 0.082), in every LR-depth stratum including genes with > 1,000 LR reads (0.001 → 0.004), so the SR contribution is biased rather than noisy. The SR generation itself matches the model: true SR read counts per transcript (recovered from RSEM read names) match truth × effective length within Poisson noise, start positions are uniform, and SR alone has the expected log-likelihood gap between MLE and truth (52k for ~81k nonzero transcripts). Gene-level SR and LR sampling rates agree with the truth within 3% across length deciles, and changing $b$ for the 100-nt poly(A) tail that the simulator's RSEM reference appends had no effect. Read-to-EC assignment is not the cause: on 500k simulated SR pairs with known origin, salmon's mapped set contains the true transcript for 99.98% of mapped reads. Convergence is not the cause either: a 30,000-iteration fit at relative tolerance 1e-4, started from the LR-only solution, reaches the same optimum (gene TV 0.082). Effective lengths are calibrated: replacing salmon's with RSEM's own, or with $l_t^{0.8}$ or $l_t^{1.2}$, is worse.

The cause is the simulation. The SR reads behave like a sample from the truth: the realized isoform fractions (true-origin counts over effective length) have gene TV 0.0173 against the truth, equal to the multinomial expectation for 10M reads (0.0174). The LR reads do not. The LR-only estimate has gene TV 0.004 for genes with 200–1,000 LR reads and 0.001 above 1,000, several times below the multinomial floor (≈ 0.015 at 500 reads). So the simulated LR molecules track the truth almost deterministically, which favours any estimator that reproduces the LR counts. On this simulation, comparisons of SR+LR against LR alone are biased towards LR, and SR+LR versus SR alone or versus other LR inputs remain valid. A fair simulation needs multinomial sampling of LR molecules. Spike-ins in real data (below) avoid the issue.

Short-read input (`analyses/sim_salmon.py`): salmon selective alignment is slightly better than kallisto (SR-only log-Pearson 0.898 vs 0.893, $\psi$ 0.835 vs 0.830; salmon's own quant.sf 0.901). Rich weights make no difference here. Salmon's bias correction slightly hurts (0.889), as expected for data simulated without bias.

### 2026-10-28 WTC11, four platforms on three biosamples

ENCSR450EXF (Illumina SE100), ENCSR507JOF (PacBio poly(A)), ENCSR309IKK (PacBio CapTrap) and ENCSR392BGY (ONT direct RNA) share biosamples ENCBS944CBA, ENCBS593PKA and ENCBS474NOC. Reference GENCODE v38. SR with salmon (genome decoys), LR with minimap2 (`analyses/wtc11/align.sh`, `quant.py`, `compare.py`).

Long-read platforms agree. PB vs ONT gives $\psi$ Pearson 0.94 and gene TV 0.08 (genes with ≥ 2 isoforms, ≥ 5 TPM in both). SR agrees with none of them ($\psi$ Pearson ≈ 0.59, gene TV ≈ 0.58) but is reproducible across biosamples ($\psi$ Pearson 0.97). The SR libraries are total RNA-seq (rRNA-depleted). 55–60% of reads map to transcripts rather than genome decoys, and 73% of SR TPM goes to misc_RNA (7SK/7SL), snRNA, ribozymes and snoRNAs, against 0% in PB. Retained-intron transcripts are 10% of the protein-coding plus retained-intron TPM in SR, against 2% in PB. Every WTC11 and H1 SR experiment in the student's ENCODE set is total RNA (the others are 32-nt PE), so WTC11 has no poly(A) SR partner for its LR. salmon's sequence and positional bias correction lowers SR–LR agreement further ($\psi$ Pearson 0.44).

The truncation model slightly lowers cross-platform LR agreement (PB vs ONT $\psi$ Pearson 0.941 → 0.920). Agreement is not accuracy, since plain models on two platforms can share the same error (e.g. assigning truncated reads to short isoforms). Real-data accuracy needs spike-ins with known isoform molarities: SIRVs in the WTC11 PacBio libraries, and SIRV/sequins in SG-NEx.

### 2026-10-28 WTC11 SIRV Set 4 ground truth

The WTC11 LR libraries carry Lexogen SIRV Set 4: 69 SIRV isoforms in 7 genes (mix E0) and 15 single-isoform long SIRVs (4–12 kb), all at 0.6 fmol, plus ERCCs. The truth is therefore $\psi = 1/n$ within each SIRV gene and equal molar abundance across all 84. LR reads were remapped to GENCODE v38 plus the spike-in transcripts (built from Lexogen's sequences and correct annotation) and scored with `analyses/wtc11/sirv_eval.py` (means over the three biosamples):

| fit | SIRV gene TV median | SD log abundance, 69 isoforms | long / isoform mean abundance |
|---|---|---|---|
| PB | 0.374 | 2.05 | 2.19 |
| PB, AS ≥ 0.95 best | 0.415 | 2.38 | 2.19 |
| PB + truncation | 0.319 | 1.56 | 2.19 |
| CT | 0.217 | 1.86 | 0.033 |
| CT, AS ≥ 0.95 best | 0.222 | 2.18 | 0.033 |
| CT + truncation | 0.209 | 1.40 | 0.033 |
| ONT dRNA | 0.131 | 1.22 | 0.22 |
| ONT dRNA, AS ≥ 0.95 best | 0.206 | 1.73 | 0.22 |
| ONT dRNA + truncation | 0.123 | 0.40 | 0.22 |

On spike-ins with known molarity, best-score ties beat the 0.95 threshold on every platform, and the truncation model improves every platform (for ONT the spread of log abundances drops from 1.22 to 0.40). Long SIRVs are over-sampled 2× by PacBio poly(A), under-sampled 30× by CapTrap and 5× by ONT direct RNA, so length-dependent sampling is large and platform specific. A per-library length bias $b_t = \exp(x_t^\top w)$ (`LengthBias` in `joli/lr_bias.py`, piecewise-linear in $\log L_t$) is identifiable only when $\theta$ is shared with a library whose rates are fixed, e.g. poly(A) SR with effective lengths, or several LR protocols on one RNA sample. PacBio remains far from the truth even with truncation (TV 0.32), possibly from PCR and size selection in library prep.

### 2026-10-29 SG-NEx spike-ins: SR, three ONT protocols, joint fits

SG-NEx H9 (SIRV Set 4: equimolar isoforms and long SIRVs) and HEYA8 (sequins Mix A v2 and SIRV Set 1 mix E2, isoforms at known unequal ratios), three replicates each, poly(A) Illumina PE150 (salmon) and ONT direct RNA, direct cDNA and PCR-cDNA (minimap2, best-score ties). Scored with `analyses/sgnex/spikein_eval.py` (means over replicates; gene TV median of isoform fractions against truth, and Pearson / SD of log abundance ratio across the mix's transcripts; SIRV-4 truth is constant, so its Pearson is undefined):

| fit | SIRV-4 TV | SIRV-4 SD log | sequins TV | sequins log-Pearson | SIRV E2 TV | SIRV E2 log-Pearson |
|---|---|---|---|---|---|---|
| SR | 0.260 | 1.80 | 0.133 | 0.900 | 0.246 | 0.832 |
| SR, salmon bias correction | 0.253 | 1.83 | 0.122 | 0.894 | 0.293 | 0.825 |
| dRNA | 0.113 | 1.97 | 0.240 | 0.699 | 0.118 | 0.664 |
| dRNA + trunc | **0.100** | 1.72 | 0.243 | 0.690 | **0.076** | **0.857** |
| SR + dRNA + trunc | 0.217 | 1.78 | 0.115 | 0.896 | 0.219 | 0.838 |
| dcDNA | 0.235 | 2.12 | 0.074 | 0.947 | 0.245 | 0.579 |
| dcDNA + trunc | 0.170 | **1.44** | 0.077 | **0.948** | 0.186 | 0.837 |
| SR + dcDNA + trunc | 0.136 | 1.61 | 0.099 | 0.910 | 0.215 | 0.855 |
| PCR-cDNA | 0.363 | 2.23 | 0.120 | 0.939 | 0.254 | 0.564 |
| PCR-cDNA + trunc | 0.263 | 1.62 | 0.123 | 0.938 | 0.249 | 0.777 |
| SR + PCR-cDNA + trunc | 0.134 | 1.57 | **0.070** | 0.914 | 0.143 | 0.856 |
| SR + all LR + trunc + length | 0.175 | 1.59 | 0.101 | 0.919 | 0.206 | 0.851 |

(dRNA detects few sequins: 120 of 164 at zero, so its sequin columns reflect depth.) Adding a length-bias term to an SR + LR fit makes every mix worse (e.g. SIRV-4 TV with dcDNA 0.136 → 0.215).

Findings. (1) The truncation model improves every ONT protocol on SIRVs, most in abundance (SIRV E2 log-Pearson 0.66 → 0.86 dRNA, 0.58 → 0.84 dcDNA, 0.56 → 0.78 PCR-cDNA), replicating the WTC11 SIRV result. (2) Joint SR + LR beats both inputs for the PCR-cDNA protocol on SIRV-4 and sequins and is better than SR alone throughout, but with direct RNA (the most accurate LR here) adding SR makes isoform fractions much worse (SIRV E2 TV 0.076 → 0.219): SR has tens of times more reads and its own isoform-level bias dominates the joint likelihood. (3) The length-bias term, with SR as the fixed reference, pulls LR towards SR's own biases. Decision (2026-10-29): ONT direct RNA is dropped as an LR input (too low throughput); the LR work uses cDNA protocols (PacBio, Kinnex, ONT direct cDNA and PCR-cDNA), where joint SR + LR helps. The next modelling steps are an SR bias model that is not taken as truth (e.g. SR positional/fragment model fitted jointly, or tempering SR's likelihood by an estimated weight) and treating no platform as the fixed length reference.

### 2026-10-28 Parse single-cell processing with salmon/alevin-fry

Parse short reads go through a patched salmon 1.10.3 (`salmon alevin --splitseqV2 --rad`, patch from tealeaf) whose RAD records carry, per alignment, the score, start, end and transcript length, then a patched alevin-fry 0.16 that parses those records and dumps transcript-level ECs (t2t map) with per-UMI probability vectors. Read positions are therefore kept for a primer-specific positional model. Cells come from tealeaf's per-series recipes: each 24-nt barcode (bc3+bc2+bc1, salmon's `--splitseqV2` order) is tagged with subject, cell type and RT primer, and alevin-fry's permit list is restricted to those barcodes (`--valid-bc`). The poly(dT) and random-hexamer halves of a cell stay separate barcodes.

Reference: GENCODE v49 basic (280,000 spliced transcripts, as in tealeaf's PBMC work) plus one unspliced target per gene spanning the gene (78,691), without genome decoys because salmon's RAD mode rejects decoy indices. Comprehensive v49 has 507k transcripts, mostly new lncRNA models, and would make 150-nt reads much more ambiguous.

`joli/alevin.py` sums per-cell EC counts into pseudobulk libraries (subject × cell type × primer), pooling ECs across sublibraries by transcript set. Poly(dT) libraries get $b_t = 1$ (one 3' priming site per molecule) and random-hexamer libraries $b_t$ = effective length. The two primer libraries of a group share one unit $\theta$, so `LengthBias` can learn the hexamer library's departure from that with poly(dT) as reference. Positional ($c$) models from the RAD positions are the next step.

Checks on 200k read pairs of GSE246624 (WT mini, 2 donors): `--softclip` (for reads starting with the template-switch oligo) raises the mapping rate from 52.8% to 55.0%. Forward alignments give 76k reads under labelled barcodes against 19k reverse, consistent with read 1 being sense. Of the UMIs, 69% (poly(dT)) and 71% (random hexamer) fall in ECs of unspliced targets only, 20–23% in ECs mixing spliced and unspliced targets (exonic reads), and 8–9% in spliced-only ECs (junction reads). The unspliced-only UMIs are spread over long genes (top 15 genes hold 5%: LINC-PINT, FYN, ARHGAP15, BACH2, …), i.e. intronic reads from pre-mRNA and internal priming, not an artefact of a few loci. `alevin-fry view` cannot parse the patched records (it assumes the standard layout), so reading positions needs a RAD reader for the 5-tag records.

### 2026-10-28 RAD positions, Parse position model, first primer comparison

`rust/radpos` dumps the patched RAD records (score, start, end, transcript length per alignment) to flat arrays; counts match salmon (109,930 mapped reads on the 200k-pair subset) and alevin-fry (76,083 reads in 5,242 cells). `joli/rad.py` keeps one read per (cell, UMI), within 0.2% of alevin-fry's UMI count, and builds pseudobulk libraries keyed by binned position. `ReadPositionModel` (in `joli/lr_bias.py`) learns the read-start distribution: distance to the 3' end in bases for poly(dT), relative position for random hexamers; `fit_bias` sets $c_{et} = b_t\, q(\text{position} \mid t)$. On spliced transcripts poly(dT) reads pile up at 3' ends (18% of starts in the last tenth) while hexamer reads lean 5' (3% in the last tenth). On a toy alternative-poly(A) simulation the position model cuts the L1 error from 1.01 to 0.06.

First primer comparison (`analyses/parse/position_fit.py`, GSE246624, 5% of molecules, 21 subject × cell type units with ≥ 200 cells, 40k–580k molecules per primer library): poly(dT)-only and hexamer-only fits barely agree, even at the gene level (log-Pearson 0.27–0.40 over genes > 50 TPM). With position models, $\psi$ Pearson 0.17 → 0.25 and gene TV 0.98 → 0.95, so the comparison is dominated by something else. Two causes:

- Unspliced targets take 83–87% of the poly(dT) TPM but 28–31% of the hexamer TPM. With $b_t = 1$ for poly(dT) every intronic UMI counts as a whole pre-mRNA molecule, but intronic poly(dT) reads come mostly from internal priming at A-rich stretches along the gene body, at a rate that grows with length like hexamer priming. Unspliced targets need a length-proportional rate under poly(dT) as well, with a learned scale (internal priming events per base relative to one 3' priming per molecule), identifiable when poly(dT) and hexamer libraries share $\theta$. Planned as extra covariates in `LengthBias` (an unspliced indicator times log length).
- Depth: about 15% of UMIs are spliced-informative, so a 5% subsample leaves a few spliced molecules per gene per library. The comparison needs all molecules and genes restricted by coverage.

### 2026-10-28 Rust E-step kernels

`rust/joli_kernels` (C ABI, ctypes, no dependencies) implements the E-step over one library's flat EC layout; `JointEM(backend="auto")` uses it when built, numpy otherwise, and tests check agreement to 1e-10. On a dedicated node one E-step takes 6.8 ms against numpy's 13.1 ms (1.2M positions) and 13.7 against 23.0 ms (2.4M); the simulated SR+LR fit drops from 8.2 to 6.0 s. Splitting one library over threads loses at these sizes (a two-pass gather doubles memory traffic, and thread start-up and cold caches cost more than a 7–14 ms call), so libraries run concurrently in a thread pool (ctypes releases the GIL) and the two-pass kernel is used only from 5M positions per thread.

Profiling the MM step (cProfile, rust backend, one thread): with one unit (simulated SR+LR) a step took 34 ms, of which 18 ms in the kernel; with 16 libraries in 16 units (K = 16, T = 218k) 445 ms, of which 178 ms in the kernel and the rest dense K × T numpy work: per-library T-length temporaries for $\mu$, Z and D, masked `np.where` passes in the update, and SQUAREM vector algebra. Changes: (1) accumulate over libraries with two matrix products on stacked arrays, $Z = \theta \odot (\pi^\top G)$ and $D = (\pi \odot N/B)^\top B_{\mathrm{mat}}$, with the kernel writing $g_l$ into rows of a preallocated $G$ ($L \times T$); (2) keep $\theta$ on transcripts that occur in some library's ECs (73% of targets for the simulated SR library, 39% for a small Parse pseudobulk), with libraries keeping global indices and rates for the bias models and `run()` taking and returning full-length arrays; (3) drop the masks (inactive entries have zero numerator) and update in place. Result: 34 → 16 ms per step (one unit) and 445 → 292 ms (16 units), Python share 55% → 34% in the 16-unit case, with identical fits (simulated SR+LR, same 508 steps and log-likelihood, $\theta$ within 3e-12). Running libraries concurrently (16 libraries, dedicated node) takes the E-step from 94 to 66 ms; the kernel is memory-latency bound.

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
