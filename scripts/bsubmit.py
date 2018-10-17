#!/usr/bin/env python3
import os
import sys
import re
import datetime
from snakemake.utils import read_job_properties

# call: bsubmit {dependencies} DATADIR LOGDIR jobscript


print(sys.argv)

LOGDIR = sys.argv[-2]
DATADIR = sys.argv[-3]
jobscript = sys.argv[-1]
mo = re.match(r'(\S+)/snakejob\.\S+\.(\d+)\.sh', jobscript)
assert mo
sm_tmpdir, sm_jobid = mo.groups()
props = read_job_properties(jobscript)

# set up job name, project name
rule = props["rule"]
if props["params"].get("logid"):
    logid = props["params"]["logid"]
else:
    logid = sm_jobid
jobname = f"{rule}-{logid}"

if props["params"].get("runtime"):
    runtime = props["params"].get("runtime")
else:
    runtime = "08:00"

if props["params"].get("queue"):
    queue = props["params"].get("queue")
else:
    queue = "mpi"

if props["params"].get("jobslots"):
    jobslots = props["params"].get("jobslots")
    jobslots = jobslots + " -R 'span[hosts=1]'"
else:
    jobslots = "1"

# -E is a pre-exec command, that reschedules the job if the command fails
#   in this case, if the data dir is unavailable (as may be the case for a hot-mounted file path)
cmdline = f'bsub -J {jobname} -r -E "ls {DATADIR}" -W {runtime} -q {queue} -n {jobslots} '

# log file output - "-oo" will overwrite existing log file, "-o" would append to existing file
if props["params"].get("logfile"):
    namedir, namefile = os.path.split(props['params'].get('logfile'))
    logfilename = f"{namedir}/{jobname}-{namefile}"
else:
    logfilename = f"{LOGDIR}/{jobname}.txt"
cmdline += f" -oo {logfilename} "

# pass memory resource request to LSF
if props["params"].get("memory"):
    memory = props["params"].get("memory")
    cmdline += f' -R "rusage[mem={memory}]" -M {memory} '

cmdline += ' -R scratch ' # request nodes with access to scratch

# figure out job dependencies
dependencies = set(sys.argv[1:-3])
if dependencies:
    cmdline += " -w '{}' ".format(" && ".join(dependencies))

# the actual job
cmdline += jobscript

# the success file
cmdline += f" {sm_tmpdir}/{sm_jobid}.jobfinished "

# the part that strips bsub's output to just the job id
cmdline += r" | tail -1 | cut -f 2 -d \< | cut -f 1 -d \>"
print(cmdline)
# call the command
os.system(cmdline)
