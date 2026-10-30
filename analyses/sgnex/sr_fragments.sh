#!/bin/bash
# SG-NEx Illumina fragment alignments for the jointly fitted SR fragment model:
# salmon quant --writeMappings streamed through sam_fragments.awk (one line per
# fragment alignment: fragment, @SQ index, start, length, AS of both mates).
# The FASTQs also hold 8-nt index reads in blocks; those pairs do not map.
#
# Run: sbatch --array=0-5 analyses/sgnex/sr_fragments.sh
#SBATCH -J sgnex_srfrag
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/logs/%x.%A_%a.log
set -euo pipefail
D=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex
SALMON=/gpfs/commons/home/daknowles/knowles_lab/software/bin/salmon
AWK=/gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/sgnex/sam_fragments.awk
SAMPLES=(H9_rep2 H9_rep3 H9_rep4 HEYA8_rep1 HEYA8_rep2 HEYA8_rep3)
s=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
row=$(awk -F'\t' -v c=${s%_rep*} -v r=${s#*_rep} '$1==c && $2==r && $3=="Illumina"' $D/tasks.tsv)
IFS=$'\t' read -r cell rep proto reads1 reads2 <<< "$row"
O=$D/sr_fragments/${s}_Illumina
mkdir -p $O
$SALMON quant -i $D/ref/salmon_decoy -l A -1 ${reads1//,/ } -2 ${reads2//,/ } -p 16 --validateMappings \
  --writeMappings -o $O/salmon 2> $O/salmon.log \
  | awk -v refs=$O/refs.txt -f $AWK | gzip -1 > $O/fragments.tsv.gz
