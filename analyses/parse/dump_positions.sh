#!/bin/bash
# Dump per-alignment positions of each library's collated RAD (alevin-fry
# collate output, labelled barcodes only) with rust/radpos.
#
# Run: sbatch --array=1-<n_libraries> --export=ALL,SERIES_DIR=<dir> analyses/parse/dump_positions.sh
#SBATCH -J parse_radpos
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/parse/logs/%x.%A_%a.log
set -euo pipefail
: "${SERIES_DIR:?set SERIES_DIR}"
RADPOS=/gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/rust/radpos/target/release/radpos
lib=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SERIES_DIR/libraries.tsv | cut -f1)
$RADPOS $SERIES_DIR/$lib/permit/map.collated.rad $SERIES_DIR/$lib/rad_dump
