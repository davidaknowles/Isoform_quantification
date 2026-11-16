#!/bin/bash
# Short-read fragment alignments for one mouse-brain pseudobulk group (row SLURM_ARRAY_TASK_ID of groups.tsv,
# 1-based): primary read pairs of each cell's Tabula Muris Senis BAM (name-collated) to node-local fastq.gz,
# then salmon --writeMappings to the spliceu index, streamed through sam_fragments.awk (one line per
# fragment alignment: fragment, @SQ index, start, length, AS of both mates).
#
# Run: sbatch --array=1-<n_groups> analyses/mouse_brain/sr_group.sh <sr_dir> <ref_dir>
#   sr_dir: groups.tsv and cells.tsv (make_groups.py); outputs go to <sr_dir>/<group>/
#SBATCH -J mb_sr
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
set -euo pipefail
SR=$1
REF=$2
SAMTOOLS=/gpfs/commons/home/daknowles/venv/kinnex/bin/samtools
SALMON=/gpfs/commons/home/daknowles/knowles_lab/software/bin/salmon
AWK=/gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/sgnex/sam_fragments.awk
group=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==i {print $1}' $SR/groups.tsv)
name=$(echo "$group" | tr '|/ ' '___')
O=$SR/$name
mkdir -p $O
TMP=/scratch/$USER/mbsr_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TMP
trap 'rm -rf $TMP' EXIT
awk -F'\t' -v g="$group" '$1==g {print $2"\t"$3}' $SR/cells.tsv > $O/cells.txt
while IFS=$'\t' read -r cell bam; do
  $SAMTOOLS collate -u -O -T $TMP/col $bam \
    | $SAMTOOLS fastq -F 0x900 -1 $TMP/${cell}_R1.fq.gz -2 $TMP/${cell}_R2.fq.gz -0 /dev/null -s /dev/null -n - 2>/dev/null
done < $O/cells.txt
$SALMON quant -i $REF/salmon_spliceu -l A -1 $TMP/*_R1.fq.gz -2 $TMP/*_R2.fq.gz -p 16 --validateMappings \
  --writeMappings -o $O/salmon 2> $O/salmon.log \
  | awk -v refs=$O/refs.txt -f $AWK | gzip -1 > $O/fragments.tsv.gz
echo "$group: $(wc -l < $O/cells.txt) cells, $(zcat $O/fragments.tsv.gz | wc -l) fragment alignments"
