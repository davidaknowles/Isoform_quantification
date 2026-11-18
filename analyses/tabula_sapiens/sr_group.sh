#!/bin/bash
# Short-read fragment alignments for one Tabula Sapiens Smart-seq2 pseudobulk group (row SLURM_ARRAY_TASK_ID
# of sr/groups.tsv, 1-based): each cell's read pair is copied from the public czb-tabula-sapiens S3 bucket to
# node-local scratch, then salmon --writeMappings to the spliceu index, streamed through sam_fragments.awk
# (one line per fragment alignment), as analyses/mouse_brain/sr_group.sh.
#
# Run: sbatch --array=1-<n_groups> analyses/tabula_sapiens/sr_group.sh <ts_dir> <ref_dir>
#SBATCH -J ts_sr
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
set -euo pipefail
D=$1
REF=$2
module load awscli
SALMON=/gpfs/commons/home/daknowles/knowles_lab/software/bin/salmon
AWK=/gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/sgnex/sam_fragments.awk
group=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==i {print $1}' $D/sr/groups.tsv)
name=$(echo "$group" | tr '|/ ' '___')
O=$D/sr/$name
mkdir -p $O
TMP=/scratch/$USER/tssr_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TMP
trap 'rm -rf $TMP' EXIT
awk -F'\t' -v g="$group" '$1==g {print $2"\t"$3"\t"$4}' $D/sr/cells.tsv > $O/cells.txt
cut -f2,3 $O/cells.txt | tr '\t' '\n' \
  | xargs -P 8 -I{} aws s3 cp --no-sign-request --only-show-errors s3://czb-tabula-sapiens/{} $TMP/
$SALMON quant -i $REF/salmon_spliceu -l A -1 $TMP/*_R1.fastq.gz -2 $TMP/*_R2.fastq.gz -p 16 --validateMappings \
  --writeMappings -o $O/salmon 2> $O/salmon.log \
  | awk -v refs=$O/refs.txt -f $AWK | gzip -1 > $O/fragments.tsv.gz
echo "$group: $(wc -l < $O/cells.txt) cells, $(zcat $O/fragments.tsv.gz | wc -l) fragment alignments"
