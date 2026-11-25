#!/bin/bash
# miniQuant (v1.4.1 Linux binary) on SG-NEx: long reads alone and hybrid with the replicate's
# Illumina pairs. miniQuant maps reads to the transcriptome itself. Short reads are rf-stranded
# (salmon library type ISR); the fragment length mean and SD are taken from salmon (no bias run).
# One array task per long-read run (see common.sh, 12 tasks).
#
# Run: sbatch --array=1-12 analyses/baselines/run_miniquant.sh
#SBATCH -J bl_miniquant
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 48:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/baselines/logs/%x.%A_%a.log
set -euo pipefail
source /gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/common.sh
module load GCC/13.2.0  # libstdc++ with GLIBCXX_3.4.32 for the prebuilt binary
lr_task $SLURM_ARRAY_TASK_ID
sr_reads
meta=$D/salmon/${cell}_rep${rep}_Illumina_nobias/aux_info/meta_info.json
fl_mean=$(grep -m1 '"frag_length_mean"' $meta | tr -dc '0-9.')
fl_sd=$(grep -m1 '"frag_length_sd"' $meta | tr -dc '0-9.')
o=$B/miniquant/$name
mkdir -p $B/miniquant
miniQuant 2>&1 | grep Version || true
SECONDS=0
[ -s $o.lr/abundance.tsv ] || miniQuant quant -r $CDNA -l $reads --long_reads_library_prep cDNA-ONT \
  -t 16 --mem 80 -o $o.lr
echo "miniQuant LR $name: $SECONDS s"; SECONDS=0
[ -s $o.hybrid/abundance.tsv ] || miniQuant quant -r $CDNA -l $reads -1 $r1 -2 $r2 --long_reads_library_prep cDNA-ONT \
  --short_reads_strandness rf-stranded --short_reads_mean_fragment_length $fl_mean \
  --short_reads_sd_fragment_length $fl_sd -t 16 --mem 80 -o $o.hybrid
echo "miniQuant hybrid $name: $SECONDS s"
