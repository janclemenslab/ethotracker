#!/usr/bin/env python3
import os
import glob

# list all mp4 files
root = '/Volumes/ukme04/#Common/playback'#'Z:\#Common\chaining'#'/Volumes/ukme04/#Common/playback'#'/scratch/clemens10/playback'#
datadir = f'{root}/dat'
logdir = f'{root}/log'
resdir = f'{root}/res'

# print(datadir)
videofiles = [file for file in glob.glob(f'{resdir}/**/*tracks.h5')]
videofiles.sort()
# print(videofiles)

# build and execute command
for filename in videofiles[::-1]:
    print(filename)
    try:
        basename = os.path.splitext(os.path.basename(filename[:-10]))[0]
        basepath = os.path.splitext(filename[:-10])[0]
        # with open(os.path.splitext(filename)[0] + '.nflies', 'r') as f:
        #     nflies = f.read()

        # cmd_track = f"python -m tracker.FlyPursuitChaining {filename} --nflies {nflies} -o -t 0.25 --led_coords 0"
        commands = [f"python3 ~/Dropbox/code.py/analysis/scripts/postprocessing.py {basepath}_tracks.h5 {os.path.join(datadir,basename,basename)}_snd.log {os.path.join(resdir,basename,basename)}_spd.h5"]
        # cmd_post = f"python ~/Dropbox/code.py/analysis/scripts/postprocessing.py {basepath}.h5 {basepath}_snd.log {os.path.join(resdir,basename)}_spd.h5"
        # cmd_post = f"python -m scripts.postprocessing_chaining {basepath}.h5 {basepath}_snd.log {os.path.join(resdir,basename)}_spd.h5"
        # # print(cmd_track)
        # # os.system(cmd_track)
        #
        # print(cmd_post)
        # os.system(cmd_post)
        # print(f'cp {basepath}.h5 {resdir}')
        # os.system(f'cp {basepath}.h5 {resdir}')
        # commands = [f"cp {filename} {filename}.bak",  # copy to backup file
        #             f"ffmpeg -y -i {filename}.bak -vcodec copy {filename}",  # Vcopy to new file
        #             f"rm {filename}.bak",  # delete backup file
        #             ]

        for cmd in commands:
            print(cmd)
            os.system(cmd)

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e)
