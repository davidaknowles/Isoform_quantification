#!/bin/bash
# Align SG-NEx ONT reads (PCR cDNA, direct cDNA) to the transcriptome as BAM for the alignment-based
# baselines (oarfish, NanoCount). Settings follow the oarfish documentation (map-ont, --eqx, up to 100
# secondary alignments); output stays in read-name order, as both tools expect.
# One array task per long-read run (see common.sh, 12 tasks).
#
# Run: sbatch --array=1-12 analyses/baselines/align_bam.sh
#SBATCH -J bl_align
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=40G
#SBATCH -t 24:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/baselines/logs/%x.%A_%a.log
set -euo pipefail
source /gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/common.sh
module load minimap2/2.29 SAMtools/1.21
lr_task $SLURM_ARRAY_TASK_ID
mkdir -p $B/bam
out=$B/bam/$name.bam
[ -s $out ] && { echo "$out exists"; exit 0; }
SECONDS=0
zcat ${reads//,/ } | minimap2 -ax map-ont --eqx -N 100 -t 16 $CDNA - 2> $B/bam/$name.minimap2.log \
  | samtools view -@ 4 -b -o $out.tmp -
mv $out.tmp $out
echo "align $name: $SECONDS s"
