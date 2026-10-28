#!/bin/bash
# Map WTC11 short reads (salmon) and long reads (minimap2 to the transcriptome)
# for the three biosamples shared by ENCSR450EXF (Illumina total RNA, SE100),
# ENCSR507JOF (PacBio poly(A)), ENCSR309IKK (PacBio CapTrap) and ENCSR392BGY
# (ONT direct RNA).
#
# Run: sbatch --array=0-14 analyses/wtc11/align.sh
#   tasks 0-5: salmon (rep x {nobias,bias}); 6-14: minimap2 (rep x {pb,captrap,ont})
# LR_REF / MM_DIR override the long-read reference fasta and output folder
# (e.g. GENCODE plus SIRV Set 4 spike-in transcripts).
#SBATCH -J wtc11_aln
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/wtc11/logs/%x.%A_%a.log
set -euo pipefail
D=/gpfs/commons/home/atalukder/RNA_Splicing/data/ENCODE_2026
OUT=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/wtc11
REF=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/ref
SALMON=/gpfs/commons/home/daknowles/knowles_lab/software/bin/salmon
MM=/nfs/sw/easybuild/software/minimap2/2.29/minimap2
BS=(ENCBS944CBA ENCBS593PKA ENCBS474NOC)
declare -A SR=([ENCBS944CBA]="ENCFF365HXR ENCFF492PIB" [ENCBS593PKA]="ENCFF621NOL ENCFF738DHG" [ENCBS474NOC]="ENCFF235BRN ENCFF742FDJ")
declare -A PB=([ENCBS944CBA]=ENCFF563QZR [ENCBS593PKA]=ENCFF370NFS [ENCBS474NOC]=ENCFF245IPA)
declare -A CT=([ENCBS944CBA]=ENCFF105WIJ [ENCBS593PKA]=ENCFF212HLP [ENCBS474NOC]=ENCFF003QZT)
declare -A ONT=([ENCBS944CBA]=ENCFF155CFF [ENCBS593PKA]=ENCFF771DIX [ENCBS474NOC]=ENCFF600LIU)
i=${SLURM_ARRAY_TASK_ID}
LR_REF=${LR_REF:-$REF/gencode.v38.transcripts.fa}
MM_DIR=${MM_DIR:-$OUT/minimap2}
mkdir -p $OUT/salmon $MM_DIR
if [ $i -lt 6 ]; then
  rep=$((i / 2)); bs=${BS[$rep]}; mode=$([ $((i % 2)) -eq 0 ] && echo nobias || echo bias)
  reads=""; for f in ${SR[$bs]}; do reads="$reads $(ls $D/SR/ENCSR450EXF/replicate_*_$bs/reads/$f.fastq.gz)"; done
  extra=""; [ $mode = bias ] && extra="--seqBias --posBias"
  $SALMON quant -i $REF/salmon_gencode38_decoy -l A -r $reads -p 16 --validateMappings \
    --dumpEq --dumpEqWeights --useEM $extra -o $OUT/salmon/${bs}_$mode
else
  j=$((i - 6)); rep=$((j / 3)); bs=${BS[$rep]}; plat=$(echo pb captrap ont | cut -d' ' -f$((j % 3 + 1)))
  case $plat in
    pb) f=$(ls $D/LR/ENCSR507JOF/replicate_*_$bs/reads/${PB[$bs]}.fastq.gz); preset="-x map-hifi" ;;
    captrap) f=$(ls $D/LR/ENCSR309IKK/replicate_*_$bs/reads/${CT[$bs]}.fastq.gz); preset="-x map-hifi" ;;
    ont) f=$(ls $D/LR/ENCSR392BGY/replicate_*_$bs/reads/${ONT[$bs]}.fastq.gz); preset="-x map-ont" ;;
  esac
  $MM -c $preset -N 100 -t 16 $LR_REF $f 2> $MM_DIR/${bs}_$plat.log \
    | gzip -1 > $MM_DIR/${bs}_$plat.paf.gz
fi
