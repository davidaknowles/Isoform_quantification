#!/bin/bash
# IGVF B01_13H mouse cortex Parse Evercode WT v2 short reads (both Illumina runs of the sublibrary): patched salmon
# 1.10.3 `alevin --splitseqV2 --rad` to the GENCODE vM37 spliced + unspliced targets, alevin-fry knee permit list
# (forward reads; cells are called before oligo-dT/random-hexamer barcodes of one well are merged), collate, and
# transcript-level quant with equivalence classes dumped (as analyses/parse/align.sh).
#
# Run: sbatch analyses/igvf_cortex/sr_align.sh <igvf_dir> <ref_dir>
#SBATCH -J igvf_sr
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
set -euo pipefail
D=$1
REF=$2
SALMON=/gpfs/commons/home/daknowles/projects/salmon-v1.10.3/build/src/salmon
FRY=/gpfs/commons/home/daknowles/projects/alevin-fry/target-portable/release/alevin-fry
T=${SLURM_CPUS_PER_TASK:-16}
[ -s $REF/t2t.tsv ] || grep '^>' $REF/spliceu.fa | sed 's/^>//; s/ .*//' | awk '{print $1"\t"$1}' > $REF/t2t.tsv
O=$D/sr_quant
mkdir -p $O
R1="$D/sr/IGVFFI2206OHYA.fastq.gz $D/sr/IGVFFI7638IJFU.fastq.gz"
R2="$D/sr/IGVFFI8607JKQU.fastq.gz $D/sr/IGVFFI8619SLHX.fastq.gz"
if [ ! -s $O/salmon_rad/map.rad ]; then
  $SALMON alevin -i $REF/salmon_spliceu -l A -1 $R1 -2 $R2 --splitseqV2 --rad --softclip -p $T -o $O/salmon_rad
fi
if [ ! -s $O/permit/map.collated.rad ]; then
  $FRY generate-permit-list -i $O/salmon_rad -d fw -k -o $O/permit
  $FRY collate -r $O/salmon_rad -i $O/permit -t $T
fi
$FRY quant -i $O/permit -m $REF/t2t.tsv -o $O/quant -t $T -r parsimony-em --use-mtx --dump-eqclasses
echo "cells: $(wc -l < $O/quant/alevin/quants_mat_rows.txt)"
