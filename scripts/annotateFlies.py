#!/usr/bin/env python3
import cv2
import os
import glob
from tracker.VideoReader import VideoReader
# list all mp4 files

root = '/Volumes/ukme04/#Common/chaining'#'/scratch/clemens10/playback'
datadir = f'{root}/dat'

print(f'{datadir}/**/*.avi')
videofiles = [file for file in glob.glob(f'{datadir}/**/*.avi')]
videofiles.sort()

# build and execute command
for videofile in videofiles:
    print(videofile)
    savefilename = os.path.splitext(videofile)[0] + '.nflies'
    if not os.path.exists(savefilename):
        vr = VideoReader(videofile)
        ret, frame = vr.read()
        if ret:
            cv2.imshow(videofile, frame)
            cv2.waitKey(1)
            nflies = input('How many flies? ')
            with open(savefilename, 'w') as f:
                f.write(nflies)
        vr.close()
