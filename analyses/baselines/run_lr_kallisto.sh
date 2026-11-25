#!/bin/bash
# lr-kallisto (kallisto 0.52.0 LongKmer build, bustools 0.45.1) on SG-NEx ONT reads, following the
# lr-kallisto workflow: pseudoalign with `kallisto bus --long` (k=63 index from prep_refs.sh,
# unmapped k-mer threshold 0.8), sort and count the BUS file into transcript-compatibility counts,
# then run the long-read EM (`kallisto quant-tcc --long -P ONT`) with the read-length distribution
# from bus.
# One array task per long-read run (see common.sh, 12 tasks).
#
# Run: sbatch --array=1-12 analyses/baselines/run_lr_kallisto.sh
#SBATCH -J bl_lrk
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/baselines/logs/%x.%A_%a.log
set -euo pipefail
source /gpfs/commons/home/daknowles/projects/JOLI/Isoform_quantification/analyses/baselines/common.sh
lr_task $SLURM_ARRAY_TASK_ID
IDX=$REF/kallisto/index_k63.idx
o=$B/lr-kallisto/$name
mkdir -p $o
kallisto version; bustools version
SECONDS=0
kallisto bus -t 16 --long --threshold 0.8 -x bulk -i $IDX -o $o ${reads//,/ }
bustools sort -t 16 $o/output.bus -o $o/sorted.bus
bustools count $o/sorted.bus -t $o/transcripts.txt -e $o/matrix.ec -o $o/count --cm -m -g $REF/t2g.tsv
flens=""; [ -s $o/flens.txt ] && flens="-f $o/flens.txt"
kallisto quant-tcc -t 16 --long -P ONT $flens --matrix-to-files $o/count.mtx -i $IDX -e $o/count.ec.txt -o $o
echo "lr-kallisto $name: $SECONDS s"
ls -la $o
