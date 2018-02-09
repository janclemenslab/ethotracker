DATADIR="/scratch/clemens10/playback"
CODEDIR="/scratch/clemens10/playback/analysis"
LOGDIR="/scratch/clemens10/playback/log"
snakemake --timestamp --rerun-incomplete --notemp --keep-going --nolock \
    --jobs 999 --immediate-submit \
    --directory . \
    --jobscript "$CODEDIR/jobscript.sh" \
    --cluster "$CODEDIR/bsubmit.py {dependencies} $DATADIR $LOGDIR" \
    "$@"