#!/usr/bin/env python3
import os
import glob
# list all mp4 files
root = '/Volumes/ukme04/#Common/playback'#'/scratch/clemens10/playback'
datadir = f'{root}/dat'
logdir = f'{root}/log'
resdir = f'{root}/playback/res'

print(f'{datadir}/**/*.mp4')
videofiles = [file for file in glob.glob(f'{datadir}/**/*.mp4')]
videofiles.sort()
print(videofiles)

# build and execute command
for videofile in videofiles:
    basename = os.path.splitext(os.path.basename(videofile))[0]
    cmd_track = f"python3 -m tracker.FlyPursuit {videofile} -t 0.3 -o --led_coords 0"
    cmd_post = f"python3 ~/analysis/scripts/postprocessing.py {basename}.h5 {basename}_snd.log {os.path.join(resdir,basename)}_spd.h5"
    # cmd = f'scripts/bsub.py "{cmd_track};{cmd_post}" --logdir {logdir} --datadir {datadir} --jobname {basename}'
    # print(cmd)
    try:
        print(cmd_track)
        os.system(cmd_track)

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e)

    try:
        print(cmd_post)
        os.system(cmd_post)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e)
