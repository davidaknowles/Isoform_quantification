# Shared paths and task lookup for the SG-NEx baseline scripts (sourced, not run).
# Long-read tasks are the ONT cDNA and direct cDNA rows of sgnex/tasks.tsv (12 rows:
# H9 rep2-4 and HEYA8 rep1-3); array index i selects the i-th such row.
D=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex
B=$D/baselines
T=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/tools/bin
REF=$D/ref
CDNA=$REF/hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa
GTF=$REF/hg38_sequins_SIRV_ERCCs_longSIRVs_v5_reformatted.gtf
GENOME=$REF/genome/hg38_sequins_SIRV_ERCCs_longSIRVs.fa
VENV=$HOME/venv/baselines
export PATH=$T:$PATH

lr_task() {  # sets cell rep proto reads for long-read task $1 (1-based)
  local row
  row=$(awk -F'\t' '$3=="cDNA" || $3=="directcDNA"' $D/tasks.tsv | sed -n "${1}p")
  IFS=$'\t' read -r cell rep proto reads _ <<< "$row"
  name=${cell}_rep${rep}_${proto}
}

sr_reads() {  # sets r1 r2 for the Illumina run of $cell $rep
  local row
  row=$(awk -F'\t' -v c=$cell -v r=$rep '$1==c && $2==r && $3=="Illumina"' $D/tasks.tsv)
  IFS=$'\t' read -r _ _ _ r1 r2 <<< "$row"
}
