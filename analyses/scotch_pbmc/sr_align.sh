#!/bin/bash
# SCOTCH Sample8 Parse WT Mini v2 Illumina reads (already demultiplexed): patched salmon 1.10.3 alevin with a custom
# geometry, barcode read = synthetic bc1+bc2+bc3 (24 nt) + UMI (prepare.py), cDNA = the public reads (sense), to the
# GENCODE v49 spliceu targets; alevin-fry knee permit list, collate, quant (as analyses/igvf_cortex/sr_align.sh).
#
# Run: sbatch analyses/scotch_pbmc/sr_align.sh <sample_dir> <ref_dir> <cdna fastq.gz>...
#SBATCH -J scotch_sr
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
set -euo pipefail
D=$1; REF=$2; shift 2
SALMON=/gpfs/commons/home/daknowles/projects/salmon-v1.10.3/build/src/salmon
FRY=/gpfs/commons/home/daknowles/projects/alevin-fry/target-portable/release/alevin-fry
T=${SLURM_CPUS_PER_TASK:-16}
O=$D/sr_quant
mkdir -p $O
if [ ! -s $O/salmon_rad/map.rad ]; then
  $SALMON alevin -i $REF/salmon_spliceu -l A -1 $D/barcodes.fastq.gz -2 "$@" \
    --bc-geometry '1[1-24]' --umi-geometry '1[25-34]' --read-geometry '2[1-end]' --rad --softclip -p $T -o $O/salmon_rad
fi
if [ ! -s $O/permit/map.collated.rad ]; then
  $FRY generate-permit-list -i $O/salmon_rad -d fw -k -o $O/permit
  $FRY collate -r $O/salmon_rad -i $O/permit -t $T
fi
$FRY quant -i $O/permit -m $REF/t2t.tsv -o $O/quant -t $T -r parsimony-em --use-mtx --dump-eqclasses
echo "cells: $(wc -l < $O/quant/alevin/quants_mat_rows.txt)"
