#!/bin/bash
# Per-library Parse processing: patched salmon 1.10.3 `alevin --splitseqV2 --rad`
# (RAD records carry alignment score, start, end and transcript length), then
# alevin-fry permit list restricted to the library's labelled barcodes,
# collate, and transcript-level quant (t2t map) with equivalence classes dumped.
# Read 1 is sense to the RNA, so only forward alignments are kept. --softclip
# lets reads that start with the template-switch oligo align.
#
# Run: sbatch --array=1-<n_libraries> --export=ALL,SERIES_DIR=<dir> analyses/parse/align.sh
#   SERIES_DIR holds libraries.tsv and <library>/valid_bc.txt (make_libraries.py)
#SBATCH -J parse_aln
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/parse/logs/%x.%A_%a.log
set -euo pipefail
: "${SERIES_DIR:?set SERIES_DIR}"
REF=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref/gencode49
SALMON=/gpfs/commons/home/daknowles/projects/salmon-v1.10.3/build/src/salmon
FRY=/gpfs/commons/home/daknowles/projects/alevin-fry/target-portable/release/alevin-fry
T=${SLURM_CPUS_PER_TASK:-16}
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SERIES_DIR/libraries.tsv)
IFS=$'\t' read -r lib read1 read2 <<< "$row"
D=$SERIES_DIR/$lib
if [ ! -s $D/salmon_rad/map.rad ]; then
  $SALMON alevin -i $REF/salmon_spliceu_decoy -l A -1 ${read1//,/ } -2 ${read2//,/ } \
    --splitseqV2 --rad --softclip -p $T -o $D/salmon_rad
fi
if [ ! -s $D/permit/map.collated.rad ]; then
  $FRY generate-permit-list -i $D/salmon_rad -d fw --valid-bc $D/valid_bc.txt -o $D/permit
  $FRY collate -r $D/salmon_rad -i $D/permit -t $T
fi
$FRY quant -i $D/permit -m $REF/t2t.tsv -o $D/quant -t $T -r parsimony-em --use-mtx --dump-eqclasses
