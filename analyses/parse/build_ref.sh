#!/bin/bash
# Spliced + unspliced (spliceu) salmon index for Parse single-cell reads:
# GENCODE v49 basic spliced transcripts, plus one unspliced target per gene
# spanning the gene ("<gene_id>-U"), with the genome as decoys. Also writes
# the transcript-to-transcript map for alevin-fry (each spliced transcript and
# each unspliced target is its own feature), a t2g table, and the spliced
# transcript fasta for aligning long reads to the same transcripts.
#
# Run: sbatch analyses/parse/build_ref.sh
#SBATCH -J parse_ref
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref/gencode49/build_ref.%j.log
set -euo pipefail
OUT=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref/gencode49
G=/gpfs/commons/home/daknowles/knowles_lab/index/hg38
GTF=$G/gencode.v49.basic.annotation.gtf.gz
GFFREAD=/gpfs/commons/home/daknowles/knowles_lab/software/bin/gffread
SALMON=/gpfs/commons/home/daknowles/projects/salmon-v1.10.3/build/src/salmon
cd $OUT
zcat $GTF > basic.gtf
$GFFREAD -w spliced.fa -g $G/hg38.fa basic.gtf
# one single-exon "transcript" per gene over the gene span
awk -F'\t' 'BEGIN{OFS="\t"} $3=="gene" {
  match($9, /gene_id "[^"]+"/); gid = substr($9, RSTART+9, RLENGTH-10)
  attr = "gene_id \"" gid "\"; transcript_id \"" gid "-U\";"
  print $1, $2, "transcript", $4, $5, ".", $7, ".", attr
  print $1, $2, "exon", $4, $5, ".", $7, ".", attr }' basic.gtf > unspliced.gtf
$GFFREAD -w unspliced.fa -g $G/hg38.fa unspliced.gtf
cat spliced.fa unspliced.fa > spliceu.fa
grep '^>' spliceu.fa | cut -c2- | cut -d' ' -f1 | awk 'BEGIN{OFS="\t"} {print $1, $1}' > t2t.tsv
awk -F'\t' '$3=="transcript"' basic.gtf | grep -o 'gene_id "[^"]*"; transcript_id "[^"]*"' \
  | sed 's/gene_id "\([^"]*\)"; transcript_id "\([^"]*\)"/\2\t\1/' > t2g_spliced.tsv
awk -F'\t' 'BEGIN{OFS="\t"} $3=="transcript" {match($9, /gene_id "[^"]+"/); g = substr($9, RSTART+9, RLENGTH-10); print g "-U", g}' unspliced.gtf > t2g_unspliced.tsv
cat t2g_spliced.tsv t2g_unspliced.tsv > t2g.tsv
cat spliceu.fa $G/hg38.fa | gzip -1 > /scratch/gentrome_parse_$SLURM_JOB_ID.fa.gz
$SALMON index -t /scratch/gentrome_parse_$SLURM_JOB_ID.fa.gz -d /gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref/hg38.decoys.txt \
  -i salmon_spliceu_decoy -k 31 --keepDuplicates -p 16
mkdir -p /scratch/$USER && mv /scratch/gentrome_parse_$SLURM_JOB_ID.fa.gz /scratch/$USER/
echo "spliced $(grep -c '^>' spliced.fa) unspliced $(grep -c '^>' unspliced.fa)"
