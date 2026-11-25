#!/bin/bash
# Build the references the SG-NEx baselines need (one-time):
#  - genome FASTA for IsoQuant: the SG-NEx spike-in contigs (chrIS sequins decoy chromosome and
#    SIRVomeERCCome), fetched by HTTP range from the SG-NEx genome FASTA using its .fai offsets, plus
#    the Ensembl GRCh38 primary assembly (same contig names as the SG-NEx GTF)
#  - IsoQuant gene database from the SG-NEx GTF (make_complete_gtf.py)
#  - kallisto index with k=63 on the SG-NEx transcriptome (lr-kallisto recommendation)
# Tools are installed by setup_tools.sh.
#
# Run: sbatch analyses/baselines/prep_refs.sh
#SBATCH -J bl_prep
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/baselines/logs/%x.%j.log
set -euo pipefail
source /gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/common.sh
module load SAMtools/1.21
HUMAN=/gpfs/commons/groups/knowles_lab/Karin/genome_files/Homo_sapiens.GRCh38.dna.primary_assembly.fa
mkdir -p $REF/genome $REF/isoquant_db $REF/kallisto

if [ ! -s $REF/kallisto/index_k63.idx ]; then
  /usr/bin/time -v kallisto index -k 63 -t 16 -i $REF/kallisto/index_k63.idx $CDNA
fi

if [ ! -s $GENOME.fai ]; then
  # chrIS and SIRVomeERCCome are the first two (single-line) records of the SG-NEx genome FASTA
  [ -s $REF/genome/spikein_contigs.fa ] || curl -s -r 0-11106273 -o $REF/genome/spikein_contigs.fa \
    https://sg-nex-data.s3.amazonaws.com/data/annotations/genome_fasta/hg38_sequins_SIRV_ERCCs_longSIRVs.fa
  cat $REF/genome/spikein_contigs.fa $HUMAN > $GENOME
  samtools faidx $GENOME
fi

if [ ! -s $REF/isoquant_db/annotation.db ]; then
  # the v5_reformatted GTF repeats transcript records per exon; use the corrected GTF (same exons) plus gene records
  source $VENV/bin/activate
  python /gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/make_complete_gtf.py \
    $D/raw/hg38_sequins_SIRV_ERCCs_longSIRVs_corrected.gtf $REF/isoquant_db/annotation.gtf
  python -c "import sys; sys.path.insert(0, '$(dirname $T)/dl/isoquant/IsoQuant-4.1.0');
from isoquant_lib.gtf2db import gtf2db; gtf2db('$REF/isoquant_db/annotation.gtf', '$REF/isoquant_db/annotation.db', complete_db=True)"
fi

ls -la $REF/genome $REF/isoquant_db $REF/kallisto
