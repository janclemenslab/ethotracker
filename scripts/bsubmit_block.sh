#!/bin/sh
export MKL_NUM_THREADS=1
export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=1"
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1
export MKL_DYNAMIC="FALSE"
export OMP_DYNAMIC="FALSE"
SCRIPTDIR="/usr/users/clemens10/analysis/scripts"
DATADIR="/scratch/clemens10/"
LOGDIR="/scratch/clemens10/log"
snakemake --rerun-incomplete --keep-going --nolock --notemp \
    --jobs 999 \
    --directory . \
    --jobscript "$SCRIPTDIR/jobscript.sh" \
    --cluster "python $SCRIPTDIR/bsubmit.py {dependencies} $DATADIR $LOGDIR" \
    -s "$@"
