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
4. Structured prior. Factor $\theta_{st} = g_{s,\gamma(t)}\, \phi_{st}$, where $g_{s\gamma}$ is the expression of gene $\gamma$ and $\phi_{s}$ holds within-gene isoform fractions. Use $\phi_{s,\gamma} \sim \mathrm{Dir}(\kappa_\gamma m_\gamma)$, with mean usage $m_\gamma$ and a gene-level precision $\kappa_\gamma$ that is shrunk across genes. Gene levels then vary freely across cell states while isoform usage is shared.
5. Flexible matching. Introduce latent units $k$ (cell states, metacells) with $\theta_k$. Each library $l$ (SR or LR, a cell, metacell or pseudobulk) is a mixture $\sum_k \pi_{lk}\theta_k$ with $\pi_{lk}$ known (same cells, shared barcodes), soft (cell-type composition) or estimated. Tie fine states to coarse ones with a tree prior, $\phi_k \sim \mathrm{Dir}(\kappa\, \phi_{\mathrm{parent}(k)})$, so fine states with only SR borrow LR information from coarser ancestors. A continuous version would regress logit isoform usage on a cell embedding.
6. Single-cell specifics. Account for UMIs and for SR chemistry: 10x 3' reads carry mostly polyA-site information, while 5' kits and Smart-seq3 carry more. Use lr-kallisto/bustools barcode handling. Batch the EM on GPU (torch sparse) across thousands of libraries.
7. Simulation. Extend `Simulations/` to generate cell states with planted DTU, and SR/LR designs with mismatched resolution (e.g. SR per fine state, LR per coarse type).
8. Baselines. lr-kallisto, oarfish, IsoQuant, Bambu/Bambu-Clump, MPAQT, miniQuant, Isosceles (single-cell pseudobulk).
9. Housekeeping. Export the conda env as a spec file, turn `JOLI_Kallisto/core` into an importable package (replacing the `sys.path` inserts), and retire `AT_code` once JK covers VI.
