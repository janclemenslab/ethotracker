import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from videoreader import VideoReader
import tracker.foreground as fg
import tracker.tracker as tk
import os
import h5py

def cropit(foreground, mask, bounding_box):
    chamber_slices = (np.s_[bounding_box[0, 0]:bounding_box[1, 0],
                            bounding_box[0, 1]:bounding_box[1, 1]])
    return foreground[chamber_slices] * mask[chamber_slices]  # crop frame to current chamber, chb+1 since 0 is background



plt.ion()

# load video
root = '/Volumes/ukme04/#Common/chaining/dat'
recname = 'localhost-20180226_161811'#'localhost-20180213_132850'#localhost-20180207_112546'  # 'localhost-20180213_145725'
filename = os.path.join(root, recname + '/' + recname + '.avi')
# filename = os.path.join(root, recname + '.avi')
print(filename)
vr = VideoReader(filename)

# load tracker results
root = '/Volumes/ukme04/#Common/chaining/res'
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
frame_number = 4000
frame_offset = 0#-120
plt.gcf().set_size_inches(20, 20)
vr.read(frame_offset)


chamber_number = 0
fly = 1
while True:
    ret, frame = vr.read(frame_number)
    frame_number +=1
    this_lines = np.clip(np.uint(lines[frame_number + frame_offset, chamber_number, :, :, :]), 0, 10000)
    frame = cropit(frame[:,:,0], chamber_mask, bounding_box)
    frame = np.stack((frame, frame, frame), axis=-1)
    frame[frame==0] = 255
    frame_with_tracks = fg.annotate(frame / 255,
                                    centers=this_lines[:,1,::-1],
                                    lines=this_lines)
    fg.show(frame_with_tracks)
    # plt.cla()
    # plt.imshow(frame_with_tracks)
    # plt.title(frame_number)
    # plt.pause(0.000001)
