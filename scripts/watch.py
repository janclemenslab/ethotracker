import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from videoreader import VideoReader
import tracker.foreground as fg
import tracker.tracker as tk
import os
import h5py
from collections import namedtuple


def cropit(foreground, mask, bounding_box):
    chamber_slices = (np.s_[bounding_box[0, 0]:bounding_box[1, 0],
                            bounding_box[0, 1]:bounding_box[1, 1]])
    return foreground[chamber_slices] * mask[chamber_slices]  # crop frame to current chamber, chb+1 since 0 is background

# plt.ion()


# load video
root = '/Volumes/ukme04/#Common/chaining/dat'
test_case = namedtuple('test_case', 'recname, frame_number, offset')
test_cases = list()
test_cases.append(test_case('localhost-20180222_130351', 63000, 0))
test_cases.append(test_case('localhost-20180226_161811', 2400, 0))
test_cases.append(test_case('localhost-20180426_105325', int(12*100*60), -250))
test_cases.append(test_case('localhost-20180424_151906', int(10*100*60), -250))
test_cases.append(test_case('localhost-20180420_151450', 48200, -250))
tst = -1
recname = test_cases[tst].recname#
frame_offset = test_cases[tst].offset
frame_number = test_cases[tst].frame_number
filename = os.path.join(root, recname + '/' + recname + '.avi')
print(filename)
vr = VideoReader(filename)

# load tracker results
root = '/Volumes/ukme04/#Common/chaining/res'
filename = os.path.join(root, recname + '_spd.h5')
with h5py.File(filename, 'r') as r:
    lines = r['lines_fixed'][:]

filename = os.path.join(root, recname + '.h5')
with h5py.File(filename, 'r') as f:
    chamber_mask = f['chambers'][:]
    chamber_bounding_box = f['chambers_bounding_box'][:]
chamber_number = 0
bounding_box = chamber_bounding_box[chamber_number+1]

# vr.frame_rate = 40
# frame_number = 66300#2400 #77850 #339850  #16500

# frame_offset = 0  # -120
frame_step = 1
plt.gcf().set_size_inches(20, 20)
vr.read(frame_offset)


frame_size = tuple(np.uint(16 * np.floor(np.array(vr[0].shape[0:2], dtype=np.double) / 16)))
# logging.warn('since x264 frame size need to be multiple of 16, frames will be truncated from {0} to {1}'.format(vr[0].shape[0:2], frame_size))
print(filename[0:-4] + "tracks.avi")

vw = cv2.VideoWriter(filename[0:-4] + "tracks.avi", fourcc=cv2.VideoWriter_fourcc(*'X264'),
                     fps=vr.frame_rate, frameSize=frame_size)

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
                                    lines=tl)
    # fg.show(frame_with_tracks)
    # vw.write(np.uint8(frame_with_tracks*255))

    plt.cla()
    plt.imshow(frame_with_tracks)
    plt.title(frame_number)
    plt.xlim(0,500)
    plt.ylim(0,500)
    plt.pause(0.000001)
# vw.release()
