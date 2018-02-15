#!/usr/bin/env python3
import os
import sys
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('jobcmd', type=str, help='command to submit')
parser.add_argument('-t', '--runtime', type=str, default='4:00', help='run time of job HH:MM')
parser.add_argument('-n', '--jobname', type=str, default='bsubjob', help='unique name of job - basename for log files')
parser.add_argument('--datadir', type=str, default='/scratch/clemens10/playback/dat/', help='log dir')
parser.add_argument('--logdir', type=str, default='/scratch/clemens10/playback/log/', help='log dir')
parser.add_argument('--dryrun', action='store_true', help='just compose the bsub-command')

args = parser.parse_args()

# -E is a pre-exec command, that reschedules the job if the command fails
#   in this case, if the data dir is unavailable (as may be the case for a hot-mounted file path)
cmdline = f'bsub -J {args.jobname} -r -E "ls {args.datadir}" -W {args.runtime} '

# log file output
cmdline += f'-oo {args.logdir}/LSF-{args.jobname}.txt '

# # pass memory resource request to LSF
# cmdline += '-R "rusage[mem={}]" -M {} '.format(mem, 2 * int(mem))

# request nodes with access to scratch
cmdline += '-R scratch '

# # figure out job dependencies
# # dependencies = set(sys.argv[1:-3])
# if dependencies:
#     cmdline += "-w '{}' ".format(" && ".join(dependencies))

# the actual job
cmdline += args.jobcmd

# # the part that strips bsub's output to just the job id
# cmdline += r" | tail -1 | cut -f 2 -d \< | cut -f 1 -d \>"
print(cmdline)
# call the command
if not args.dryrun:
    os.system(cmdline)
