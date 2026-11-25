#!/bin/bash
# Install the baseline quantifiers used on SG-NEx (one-time). Prebuilt binaries go to
# $TOOLS/dl/<tool>/ with links in $TOOLS/bin; Python tools go to the uv venv ~/venv/baselines;
# MPAQT (R package) goes to the R library $TOOLS/Rlib for module R/4.4.1.
#   oarfish 0.13.0, kallisto 0.52.0 (LongKmer build, k up to 63), bustools 0.45.1,
#   miniQuant 1.4.1 (needs GCC/13.2.0 runtime), NanoCount 1.1.0.post2, IsoQuant 4.1.0, MPAQT 2.4.0
# The R part compiles gpboost, so run this script through Slurm.
#
# Run: sbatch analyses/baselines/setup_tools.sh
#SBATCH -J bl_setup
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /gpfs/commons/groups/knowles_lab/daknowles/JOLI/sgnex/baselines/logs/%x.%j.log
set -euo pipefail
TOOLS=/gpfs/commons/groups/knowles_lab/daknowles/JOLI/tools
GH=https://github.com
mkdir -p $TOOLS/bin $TOOLS/dl/{oarfish,kallisto,bustools,miniquant,isoquant} $TOOLS/Rlib
cd $TOOLS/dl/oarfish
[ -x oarfish-x86_64-unknown-linux-gnu/oarfish ] || curl -sL $GH/COMBINE-lab/oarfish/releases/download/v0.13.0/oarfish-x86_64-unknown-linux-gnu.tar.xz | tar xJ
cd $TOOLS/dl/kallisto
[ -x kallisto/kallisto ] || curl -sL $GH/pachterlab/kallisto/releases/download/v0.52.0/kallisto_linux_LongKmer-v0.52.0.tar.gz | tar xz
cd $TOOLS/dl/bustools
[ -x bustools/bustools ] || curl -sL $GH/BUStools/bustools/releases/download/v0.45.1/bustools_linux-master.tar.gz | tar xz
cd $TOOLS/dl/miniquant
[ -x miniQuant_linux/miniQuant ] || curl -sL $GH/Augroup/miniQuant/releases/download/v1.4.1/miniQuant_linux_v1.4.1.tar.gz | tar xz
chmod +x miniQuant_linux/miniQuant
cd $TOOLS/dl/isoquant
[ -d IsoQuant-4.1.0 ] || curl -sL $GH/ablab/IsoQuant/archive/refs/tags/v4.1.0.tar.gz | tar xz
cd $TOOLS/bin
ln -sf ../dl/oarfish/oarfish-x86_64-unknown-linux-gnu/oarfish oarfish
ln -sf ../dl/kallisto/kallisto/kallisto kallisto
ln -sf ../dl/bustools/bustools/bustools bustools
ln -sf ../dl/miniquant/miniQuant_linux/miniQuant miniQuant

# Python: NanoCount and IsoQuant dependencies
[ -x ~/venv/baselines/bin/python ] || uv venv -p 3.11 ~/venv/baselines
source ~/venv/baselines/bin/activate
uv pip install NanoCount==1.1.0.post2 -r $TOOLS/dl/isoquant/IsoQuant-4.1.0/requirements.txt

# R: MPAQT from GitHub (tagged 2.4.0) into a private library
[ -d $TOOLS/dl/mpaqt ] || git clone -q $GH/csglab/MPAQT.git $TOOLS/dl/mpaqt
module load R/4.4.1
R_LIBS_USER=$TOOLS/Rlib Rscript -e "options(repos='https://cloud.r-project.org', Ncpus=8);
  .libPaths(c('$TOOLS/Rlib', .libPaths()));
  if (!requireNamespace('gpboost', quietly=TRUE)) install.packages('gpboost', lib='$TOOLS/Rlib');
  remotes::install_local('$TOOLS/dl/mpaqt', lib='$TOOLS/Rlib', dependencies=FALSE, upgrade='never');
  library(mpaqt); print(packageVersion('mpaqt'))"
