#!/bin/sh
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1

DATADIR="/scratch/clemens10/"
CODEDIR="/usr/users/clemens10/analysis/scripts"
LOGDIR="/scratch/clemens10/log"
snakemake --timestamp --rerun-incomplete --notemp --keep-going --nolock \
    --jobs 999 --immediate-submit \
    --directory . \
    --jobscript "$CODEDIR/jobscript.sh" \
    --cluster "$CODEDIR/bsubmit.py {dependencies} $DATADIR $LOGDIR" \
    "$@"
