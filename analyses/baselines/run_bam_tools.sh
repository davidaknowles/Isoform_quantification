#!/bin/bash
# Alignment-based long-read baselines on the transcriptome BAMs from align_bam.sh.
#   oarfish   (v0.13.0) two settings: the published recommendation (--filter-group no-filters
#             --model-coverage) and the learned read-placement mixture (--model-truncation).
#             (--short-quant only seeds the EM start, so no hybrid oarfish run.)
#   nanocount (NanoCount 1.1.0.post2) default filters
# One array task per long-read run (see common.sh, 12 tasks); the first argument picks the tool.
#
# Run: sbatch --array=1-12 analyses/baselines/run_bam_tools.sh oarfish
#      sbatch --array=1-12 --mem=160G -t 72:00:00 analyses/baselines/run_bam_tools.sh nanocount
#SBATCH -J bl_bam
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/baselines/logs/%x.%A_%a.log
set -euo pipefail
source /gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/common.sh
tool=$1
lr_task $SLURM_ARRAY_TASK_ID
bam=$B/bam/$name.bam
[ -s $bam ] || { echo "missing $bam"; exit 1; }
out=$B/$tool/$name
mkdir -p $B/$tool
case $tool in
  oarfish)
    oarfish --version
    SECONDS=0
    oarfish -j 16 -a $bam -o $out.cov --filter-group no-filters --model-coverage
    echo "oarfish cov $name: $SECONDS s"; SECONDS=0
    oarfish -j 16 -a $bam -o $out.trunc --filter-group no-filters --model-truncation
    echo "oarfish trunc $name: $SECONDS s"
    ;;
  nanocount)
    source $VENV/bin/activate
    NanoCount --version
    SECONDS=0
    NanoCount -i $bam -o $out.tsv -x
    echo "nanocount $name: $SECONDS s"
    ;;
esac
