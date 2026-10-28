#!/bin/bash
# Map SG-NEx reads: Illumina with salmon (rich ECs, with and without bias
# correction), long reads with minimap2 to the transcriptome.
# One array task per row of tasks.tsv (from make_tasks.py); Illumina rows run salmon twice.
#
# Run: sbatch --array=1-25 analyses/sgnex/align.sh
#SBATCH -J sgnex_aln
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/logs/%x.%A_%a.log
set -euo pipefail
D=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex
SALMON=/gpfs/commons/home/daknowles/knowles_lab/software/bin/salmon
MM=/nfs/sw/easybuild/software/minimap2/2.29/minimap2
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $D/tasks.tsv)
IFS=$'\t' read -r cell rep proto reads1 reads2 <<< "$row"
name=${cell}_rep${rep}_${proto}
mkdir -p $D/salmon $D/minimap2
if [ "$proto" = Illumina ]; then
  for mode in nobias bias; do
    extra=""; [ $mode = bias ] && extra="--seqBias --gcBias --posBias"
    $SALMON quant -i $D/ref/salmon_decoy -l A -1 ${reads1//,/ } -2 ${reads2//,/ } -p 16 --validateMappings \
      --dumpEq --dumpEqWeights --useEM $extra -o $D/salmon/${name}_$mode
  done
else
  preset="-x map-ont"; [ "$proto" = PacBio-SMRTcell ] && preset="-x map-hifi"
  zcat ${reads1//,/ } | $MM -c $preset -N 100 -t 16 $D/ref/hg38_sequins_SIRV_ERCCs_longSIRVs_cdna.fa - \
    2> $D/minimap2/$name.log | gzip -1 > $D/minimap2/$name.paf.gz
fi
