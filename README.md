# analysis

## installation
requirements...
`pip install -e .`

## organization
workflow:

organized with [snakemake](snakemake), a make-like tool
`bsubmit_block.sh`



# TODO
[x] use logging to flag different log levels (info, warning, error)
[ ] implement exceptions (e.g ObjectLostException, ...)
[ ] implement chamber abstract class finder which can either take annotation file data or detects chambers automatically
[x] implement background detector abstract class for max/min/mean/median/MOG
[x] display class...
[ ] have yaml/toml configuration file to combine all options and their parameters

# config format
function.background = "command"
function.chamber = "command"
function.trackchamber = "command"


# playback analyses
track
scripts/postprocessing.py
scripts/gather.py
