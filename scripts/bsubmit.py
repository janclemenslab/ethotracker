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
jobname = "{rule}-{jobid}".format(rule=props["rule"], jobid=sm_jobid)
if props["params"].get("logid"):
    jobname = "{rule}-{id}".format(rule=props["rule"], id=props["params"]["logid"])

if props["params"].get("runtime"):
    runtime = props["params"].get("runtime")
else:
    runtime = "08:00"

# -E is a pre-exec command, that reschedules the job if the command fails
#   in this case, if the data dir is unavailable (as may be the case for a hot-mounted file path)
cmdline = 'bsub -J {jobname} -r -E "ls {datadir}" -W {runtime} '.format(jobname=jobname, datadir=DATADIR, runtime=runtime)
# cmdline = 'bsub -J {jobname} -W {runtime}'.format(jobname=jobname, runtime="08:00")

# log file output - "-oo" will overwrite existing log file, "-o" would append to existing file
if props["params"].get("logfile"):
    logfilename = f"{jobname}-{props["params"].get("logfile")}"
else:
    logfilename = "{logdir}/{jobname}.txt".format(logdir=LOGDIR, jobname=jobname)
cmdline += f"-oo {logfilename} "

# # pass memory resource request to LSF
# mem = props.get('resources', {}).get('mem')
# if mem:
#     cmdline += '-R "rusage[mem={}]" -M {} '.format(mem, 2 * int(mem))
cmdline += '-R scratch ' # request nodes with access to scratch

# figure out job dependencies
dependencies = set(sys.argv[1:-3])
if dependencies:
    cmdline += "-w '{}' ".format(" && ".join(dependencies))

# the actual job
cmdline += jobscript

# the success file
cmdline += " %s/%s.jobfinished" % (sm_tmpdir, sm_jobid)

# the part that strips bsub's output to just the job id
cmdline += r" | tail -1 | cut -f 2 -d \< | cut -f 1 -d \>"
print(cmdline)
# call the command
os.system(cmdline)
