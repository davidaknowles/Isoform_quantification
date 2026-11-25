#!/bin/bash
# IGVF B01_13H mouse cortex ONT long reads (Parse WT v2 library), in the layout of analyses/kinnex_parse:
#   index      minimap2 map-ont index of the GENCODE vM37 spliceu targets
#   extract    one task per ONT FASTQ (files.tsv order): Parse barcodes, UMI, primer, cDNA (ont_barcodes.py)
#   align      one task per chunk: minimap2 map-ont (up to 100 secondary alignments) of the cDNA to the spliceu targets
#   sr_cells   after sr_align.sh: RAD position dump (rust/radpos), sr_cells.py (cells, B6J, gene counts) and CellTypist
#              (Mouse_Isocortex_Hippocampus) labels
#   molecules  one task per chunk: sense best-score alignments joined to called cells (kinnex_parse/molecules.py chunk)
#   combine    one read per molecule (cell, UMI, primer) and the cell x gene matrix (molecules.py combine)
#
# Run: sbatch analyses/igvf_cortex/run.sh index
#      sbatch --array=0-9 analyses/igvf_cortex/run.sh extract
#      sbatch --array=0-9 analyses/igvf_cortex/run.sh align
#      sbatch analyses/igvf_cortex/run.sh sr_cells
#      sbatch --array=0-9 analyses/igvf_cortex/run.sh molecules; sbatch analyses/igvf_cortex/run.sh combine
#SBATCH -J igvf_ont
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/igvf_13H/logs/%x.%A_%a.log
set -euo pipefail
D=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/igvf_13H
REF=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref/gencodeM37
REPO=/gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification
MM=/nfs/sw/easybuild/software/minimap2/2.29/minimap2
T=${SLURM_CPUS_PER_TASK:-16}
k=${SLURM_ARRAY_TASK_ID:-0}
kk=$(printf %03d $k)
mkdir -p $D/reads $D/minimap2 $D/molecules
case $1 in
  index)
    $MM -x map-ont -d $REF/spliceu.ont.mmi $REF/spliceu.fa ;;
  extract)
    fq=$(awk -F'\t' '$8=="ont" && $4=="fastq" {print $2}' $D/files.tsv | sed -n "$((k + 1))p")
    ~/venv/kinnex/bin/python $REPO/analyses/igvf_cortex/ont_barcodes.py $D $D $D/ont/$fq.fastq.gz --chunk $k ;;
  align)
    $MM -c -x map-ont -N 100 -t $T $REF/spliceu.ont.mmi $D/reads/cdna_$kk.fa.gz 2> $D/minimap2/cdna_$kk.log \
      | gzip -1 > $D/minimap2/cdna_$kk.paf.gz ;;
  sr_cells)
    [ -s $D/sr_quant/rad_dump/refs.txt ] || $REPO/rust/radpos/target/release/radpos $D/sr_quant/permit/map.collated.rad $D/sr_quant/rad_dump
    ~/venv/joli/bin/python $REPO/analyses/igvf_cortex/sr_cells.py $D $REF
    module load python/3.12.3-GCCcore-13.3.0
    ~/venv/celltypist/bin/python $REPO/analyses/kinnex_parse/celltypist_labels.py $D/molecules_sr $D/celltypist.tsv 0 Mouse_Isocortex_Hippocampus.pkl ;;
  molecules)
    ~/venv/joli/bin/python $REPO/analyses/kinnex_parse/molecules.py $D $REF chunk $kk ;;
  combine)
    ~/venv/joli/bin/python $REPO/analyses/kinnex_parse/molecules.py $D $REF combine ;;
esac
