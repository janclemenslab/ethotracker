
SCRIPTDIR="/usr/users/clemens10/analysis/scripts"
DATADIR="/scratch/clemens10/playback/"
LOGDIR="/scratch/clemens10/playback/log"
snakemake --timestamp --rerun-incomplete --keep-going --nolock --notemp \
    --jobs 999 \
    --directory . \
    --jobscript "$SCRIPTDIR/jobscript.sh" \
    --cluster "python $SCRIPTDIR/bsubmit.py {dependencies} $DATADIR $LOGDIR" \
    "$@"
