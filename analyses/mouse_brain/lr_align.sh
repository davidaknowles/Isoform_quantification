#!/bin/bash
# minimap2 (map-hifi, up to 100 secondary alignments) of one ENCODE PacBio long-read fastq (row
# SLURM_ARRAY_TASK_ID of files.tsv, 1-based) to the mouse spliceu transcripts; PAF with cigar.
#
# Run: sbatch --array=1-<n_files> analyses/mouse_brain/lr_align.sh <lr_dir> <ref_dir>
#SBATCH -J mb_lr
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
set -euo pipefail
LR=$1
REF=$2
MM=/nfs/sw/easybuild/software/minimap2/2.29/minimap2
row=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==i' $LR/files.tsv)
acc=$(echo "$row" | cut -f1)
fq=$(echo "$row" | cut -f7)
mkdir -p $LR/minimap2
$MM -c -x map-hifi -N 100 -t 16 $REF/spliceu.hifi.mmi $fq 2> $LR/minimap2/$acc.log | gzip -1 > $LR/minimap2/$acc.paf.gz
