import numpy as np
import pandas as pd
import cv2
from videoreader import VideoReader
import tracker.foreground as fg
import tracker.tracker as tk
import os
import h5py
import sys


def cropit(foreground, mask, bounding_box):
    chamber_slices = (np.s_[bounding_box[0, 0]:bounding_box[1, 0],
                            bounding_box[0, 1]:bounding_box[1, 1]])
    return foreground[chamber_slices] * mask[chamber_slices]  # crop frame to current chamber, chb+1 since 0 is background

# do the damage
filename = sys.argv[1]
recname = os.path.splitext(os.path.split(filename)[-1])[0]
print(recname)
# # load video
# root = '/Volumes/ukme04/#Common/chaining/dat'
# recname = 'localhost-20180411_172714'  # 'localhost-20180413_154616'#'localhost-20180213_132850'#localhost-20180207_112546'  # 'localhost-20180213_145725'
# filename = os.path.join(root, recname + '/' + recname + '.avi')
# filename = os.path.join(root, recname + '.avi')
print(filename)
vr = VideoReader(filename)

# load tracker results
root = '/scratch/clemens10/chaining/res'#'/Volumes/ukme04/#Common/chaining/res'
filename = os.path.join(root, recname + '_spd.h5')
# filename = os.path.join(root, recname + '.h5')
with h5py.File(filename, 'r') as r:
    lines = r['lines_fixed'][:]

filename = os.path.join(root, recname + '.h5')
with h5py.File(filename, 'r') as f:
    chamber_mask = f['chambers'][:]
    chamber_bounding_box = f['chambers_bounding_box'][:]
chamber_number = 0
bounding_box = chamber_bounding_box[chamber_number+1]

# vr.frame_rate = 40
frame_number = 10000  # 77850#16500
frame_offset = 0  # -120
frame_step = 1
vr.read(frame_offset)


frame_size = tuple(np.uint(16 * np.floor(np.array(vr[0].shape[0:2], dtype=np.double) / 16)))
# logging.warn('since x264 frame size need to be multiple of 16, frames will be truncated from {0} to {1}'.format(vr[0].shape[0:2], frame_size))
print(filename[0:-4] + "tracks.avi")

vw = cv2.VideoWriter(filename[0:-4] + "tracks.avi", fourcc=cv2.VideoWriter_fourcc(*'MPEG'), fps=vr.frame_rate, frameSize=frame_size)

chamber_number = 0
fly = 1
while frame_number < 340000:
    ret, frame = vr.read(frame_number)
    if frame_number % 100 == 0:
        print(frame_number)
    frame_number += frame_step
    this_lines = np.clip(np.uint(lines[frame_number + frame_offset, chamber_number, :, :, :]), 0, 10000)
    tl = this_lines.copy()
    tl[:, 1, ::-1] = this_lines[:, 1, ::-1] + bounding_box[0].T
    tl[:, 0, ::-1] = this_lines[:, 0, ::-1] + bounding_box[0].T
    tl = np.int16(tl)
    frame_with_tracks = fg.annotate(frame / 255,
                                    centers=tl[:, 1, ::-1],
                                    lines=tl)
    # fg.show(frame_with_tracks)
    vw.write(np.uint8(frame_with_tracks*255))

vw.release()
