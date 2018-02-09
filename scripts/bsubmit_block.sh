DATADIR="/scratch/clemens10/playback/"
LOGDIR="/scratch/clemens10/playback/log"
snakemake --timestamp --rerun-incomplete --keep-going --nolock --notemp \
    --jobs 999 \
    --directory . \
    --jobscript "jobscript.sh" \
    --cluster "python bsubmit.py {dependencies} $DATADIR $LOGDIR" \
    "$@"
