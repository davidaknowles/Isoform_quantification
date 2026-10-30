#!/bin/bash
# PacBio Kinnex 10x 3'/5' PBMC long reads (PacBio's UMI-deduplicated, barcode-corrected reads):
#   extract   one task per library (libraries.tsv row): dedup BAM -> 20M-read cDNA/read-table chunks
#   align     one task per chunk (chunks.txt): minimap2 to the spliceu transcripts used for Parse SR
#   molecules one task per chunk: strict sense alignments joined to cells (kinnex_parse/molecules.py)
#   combine   per-molecule alignments, cell x gene matrix, CellTypist labels
#
# Run: sbatch --array=1-7 analyses/kinnex_10x/run.sh extract
#      ls <out>/reads/cdna_*.fa.gz | sed 's/.*cdna_//; s/.fa.gz//' > <out>/chunks.txt
#      sbatch --array=1-<n> analyses/kinnex_10x/run.sh align
#      python analyses/kinnex_10x/call_cells.py <out> ; sbatch --array=1-<n> ... molecules ; sbatch ... combine
#SBATCH -J kinnex10x
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/kinnex_10x/logs/%x.%A_%a.log
set -euo pipefail
K=/gpfs/commons/groups/knowles_lab/data/sc/pacbio_kinnex_pbmc
OUT=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/kinnex_10x
REF=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref/gencode49
REPO=/gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification
SAMTOOLS=/gpfs/commons/home/daknowles/venv/kinnex/bin/samtools
MM=/nfs/sw/easybuild/software/minimap2/2.29/minimap2
T=${SLURM_CPUS_PER_TASK:-16}
mkdir -p $OUT/reads $OUT/minimap2 $OUT/molecules
case $1 in
  extract)
    row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $REPO/analyses/kinnex_10x/libraries.tsv)
    IFS=$'\t' read -r id kit cells dir <<< "$row"
    $SAMTOOLS view -@ 4 $K/$dir/2-DeduplicatedReads/scisoseq.5p--3p.tagged.refined.corrected.sorted.dedup.bam \
      | awk -v lib=$id -v lib_index=$SLURM_ARRAY_TASK_ID -v kit=$kit -v out=$OUT/reads -v chunk=20000000 \
            -f $REPO/analyses/kinnex_10x/extract.awk ;;
  align)
    k=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $OUT/chunks.txt)
    $MM -c -x map-hifi -N 100 -t $T $REF/spliceu.hifi.mmi $OUT/reads/cdna_$k.fa.gz 2> $OUT/minimap2/cdna_$k.log \
      | gzip -1 > $OUT/minimap2/cdna_$k.paf.gz ;;
  molecules)
    k=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $OUT/chunks.txt)
    ~/venv/joli/bin/python $REPO/analyses/kinnex_parse/molecules.py $OUT $REF chunk $k ;;
  combine)
    ~/venv/joli/bin/python $REPO/analyses/kinnex_parse/molecules.py $OUT $REF combine
    module load python/3.12.3-GCCcore-13.3.0
    ~/venv/celltypist/bin/python $REPO/analyses/kinnex_parse/celltypist_labels.py $OUT/molecules $OUT/celltypist.tsv ;;
esac
