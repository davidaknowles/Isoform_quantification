#!/bin/bash
# PacBio Kinnex Parse PBMC S-reads: barcode/UMI/primer calling (extract.py), then
# minimap2 alignment of the trimmed cDNA to the spliced+unspliced transcript set
# used for the Parse short reads (one array task per cDNA chunk).
#
# Run: sbatch analyses/kinnex_parse/run.sh extract
#      sbatch --array=0-<n_chunks-1> analyses/kinnex_parse/run.sh align
#SBATCH -J kinnex_parse
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/kinnex_parse/logs/%x.%A_%a.log
set -euo pipefail
BAM=/gpfs/commons/groups/knowles_lab/data/sc/pacbio_kinnex_pbmc/DATA-Revio-Kinnex-PBMC-Parse/1-Sreads/segmented.bam
RECIPE=/gpfs/commons/home/daknowles/projects/tealeaf/.cache/pbmc/GSE295527
OUT=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/kinnex_parse
REF=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref/gencode49
SAMTOOLS=/gpfs/commons/home/daknowles/venv/kinnex/bin/samtools
MM=/nfs/sw/easybuild/software/minimap2/2.29/minimap2
REPO=/gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification
T=${SLURM_CPUS_PER_TASK:-16}
case $1 in
  extract)
    $SAMTOOLS fastq -@ 4 $BAM 2> $OUT/samtools_fastq.log \
      | ~/venv/joli/bin/python $REPO/analyses/kinnex_parse/extract.py $RECIPE $OUT/reads $((T - 4)) ;;
  index)
    [ -s $REF/spliceu.hifi.mmi ] || $MM -x map-hifi -d $REF/spliceu.hifi.mmi $REF/spliceu.fa ;;
  align)
    k=$(printf %03d ${SLURM_ARRAY_TASK_ID})
    $MM -c -x map-hifi -N 100 -t $T $REF/spliceu.hifi.mmi $OUT/reads/cdna_$k.fa.gz 2> $OUT/minimap2/cdna_$k.log \
      | gzip -1 > $OUT/minimap2/cdna_$k.paf.gz ;;
  molecules)
    k=$(printf %03d ${SLURM_ARRAY_TASK_ID})
    ~/venv/joli/bin/python $REPO/analyses/kinnex_parse/molecules.py $OUT $REF chunk $k ;;
  combine)
    ~/venv/joli/bin/python $REPO/analyses/kinnex_parse/molecules.py $OUT $REF combine
    module load python/3.12.3-GCCcore-13.3.0
    ~/venv/celltypist/bin/python $REPO/analyses/kinnex_parse/celltypist_labels.py $OUT/molecules $OUT/celltypist.tsv ;;
esac
# Downstream (after align and call_cells.py):
#   sbatch --array=0-<n_chunks-1> analyses/kinnex_parse/run.sh molecules
#   sbatch analyses/kinnex_parse/run.sh combine
