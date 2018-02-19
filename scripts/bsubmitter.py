#!/usr/bin/env python3
import os
import glob


# list all mp4 files
root = 'Z:\#Common\chaining'#'/Volumes/ukme04/#Common/chaining'#'/scratch/clemens10/playback'
datadir = f'{root}/dat'
logdir = f'{root}/log'
resdir = f'{root}/playback/res'

print(datadir)
videofiles = [file for file in glob.glob(f'{datadir}/**/*.avi')]
videofiles.sort()
print(videofiles)

# build and execute command
for videofile in videofiles:
    try:
        basename = os.path.splitext(os.path.basename(videofile))[0]
        print(os.path.splitext(videofile)[0] + '.nflies',)
        with open(os.path.splitext(videofile)[0] + '.nflies', 'r') as f:
            nflies = f.read()

        cmd_track = f"python -m tracker.FlyPursuitChaining {videofile} --nflies {nflies} -o -t 0.3 --led_coords 0"
        # cmd_post = f"python3 ~/analysis/scripts/postprocessing.py {basename}.h5 {basename}_snd.log {os.path.join(resdir,basename)}_spd.h5"

        print(cmd_track)
        os.system(cmd_track)

        # print(cmd_post)
        # os.system(cmd_post)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e)