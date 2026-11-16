#!/bin/bash
# Mouse spliced + unspliced (spliceu) reference: GENCODE vM37 basic spliced transcripts on GRCm39 plus one
# unspliced target per gene spanning the gene ("<gene_id>-U"), as analyses/parse/build_ref.sh for human.
# Builds a salmon index (short-read fragment mappings) and a minimap2 map-hifi index (PacBio long reads)
# of the same targets, and t2g tables.
#
# Run: sbatch analyses/mouse_brain/build_ref.sh <annotation_dir> <out_dir>
#   annotation_dir: GRCm39.genome.fa and gencode.vM37.basic.annotation.gtf
#SBATCH -J mm_ref
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
set -euo pipefail
A=$1
OUT=$2
GFFREAD=/gpfs/commons/home/daknowles/knowles_lab/software/bin/gffread
SALMON=/gpfs/commons/home/daknowles/projects/salmon-v1.10.3/build/src/salmon
MM=/nfs/sw/easybuild/software/minimap2/2.29/minimap2
mkdir -p $OUT && cd $OUT
cp $A/gencode.vM37.basic.annotation.gtf basic.gtf
$GFFREAD -w spliced.fa -g $A/GRCm39.genome.fa basic.gtf
awk -F'\t' 'BEGIN{OFS="\t"} $3=="gene" {
  match($9, /gene_id "[^"]+"/); gid = substr($9, RSTART+9, RLENGTH-10)
  attr = "gene_id \"" gid "\"; transcript_id \"" gid "-U\";"
  print $1, $2, "transcript", $4, $5, ".", $7, ".", attr
  print $1, $2, "exon", $4, $5, ".", $7, ".", attr }' basic.gtf > unspliced.gtf
$GFFREAD -w unspliced.fa -g $A/GRCm39.genome.fa unspliced.gtf
cat spliced.fa unspliced.fa > spliceu.fa
awk -F'\t' '$3=="transcript"' basic.gtf | grep -o 'gene_id "[^"]*"; transcript_id "[^"]*"' \
  | sed 's/gene_id "\([^"]*\)"; transcript_id "\([^"]*\)"/\2\t\1/' > t2g_spliced.tsv
awk -F'\t' 'BEGIN{OFS="\t"} $3=="transcript" {match($9, /gene_id "[^"]+"/); g = substr($9, RSTART+9, RLENGTH-10); print g "-U", g}' unspliced.gtf > t2g_unspliced.tsv
cat t2g_spliced.tsv t2g_unspliced.tsv > t2g.tsv
$SALMON index -t spliceu.fa -i salmon_spliceu -k 31 --keepDuplicates -p 16
$MM -x map-hifi -d spliceu.hifi.mmi spliceu.fa
echo "spliced $(grep -c '^>' spliced.fa) unspliced $(grep -c '^>' unspliced.fa)"
