#!/bin/bash
# IsoQuant 4.1.0 on SG-NEx ONT reads: spliced alignment to the genome (minimap2, run by IsoQuant) and
# reference-transcript quantification only (--analysis quant, no transcript discovery), default
# transcript quantification strategy (unique_only). Genome and gene database come from prep_refs.sh.
# One array task per long-read run (see common.sh, 12 tasks).
#
# Run: sbatch --array=1-12 analyses/baselines/run_isoquant.sh
#SBATCH -J bl_isoquant
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 96:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/baselines/logs/%x.%A_%a.log
set -euo pipefail
source /gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/common.sh
module load minimap2/2.29 SAMtools/1.21
source $VENV/bin/activate
lr_task $SLURM_ARRAY_TASK_ID
IQ=$(dirname $T)/dl/isoquant/IsoQuant-4.1.0/isoquant.py
o=$B/isoquant/$name
mkdir -p $o
python $IQ --version
SECONDS=0
python $IQ --reference $GENOME --genedb $REF/isoquant_db/annotation.db --fastq ${reads//,/ } \
  --data_type nanopore --analysis quant --labels $name --prefix $name -t 16 -o $o --force
echo "IsoQuant $name: $SECONDS s"
