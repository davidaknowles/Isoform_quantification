#!/bin/bash
# Slurm wrapper for run_mpaqt.R (R/4.4.1, MPAQT in the tools R library, kallisto/bustools from
# tools/bin). With "rep", array index 1-6 selects H9 rep2-4, HEYA8 rep1-3; needs the lr-kallisto
# outputs of that replicate.
#
# Run: sbatch analyses/baselines/run_mpaqt.sh index
#      sbatch --array=1-6 analyses/baselines/run_mpaqt.sh rep
#SBATCH -J bl_mpaqt
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 48:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/baselines/logs/%x.%A_%a.log
set -euo pipefail
source /gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/common.sh
module load R/4.4.1
R_SCRIPT=/gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/run_mpaqt.R
SECONDS=0
if [ "$1" = index ]; then
  Rscript $R_SCRIPT index
else
  reps=(H9:2 H9:3 H9:4 HEYA8:1 HEYA8:2 HEYA8:3)
  IFS=: read -r cell rep <<< "${reps[$((SLURM_ARRAY_TASK_ID - 1))]}"
  Rscript $R_SCRIPT rep $cell $rep
fi
echo "mpaqt $*: $SECONDS s"
