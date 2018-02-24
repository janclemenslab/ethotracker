import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tracker.VideoReader import VideoReader
import tracker.ForeGround as fg
import tracker.Tracker as tk
import os
import h5py

plt.ion()

# load video
root = '/Volumes/ukme04/#Common/chaining/dat'
recname = 'localhost-20180207_112546'  # 'localhost-20180213_145725'
filename = os.path.join(root, recname + '/' + recname + '.avi')
print(filename)
vr = VideoReader(filename)

# load tracker results
filename = os.path.join(root, recname + '/' + recname + '.h5')
r = h5py.File(filename)

# vr.frame_rate = 40
frame_number = 250
frame_offset = 0
plt.gcf().set_size_inches(20, 20)
vr.read(frame_offset)

lines = np.zeros((r['lines'].shape[2], 2, 2))

chamber_number = 0
old_lines = r['lines'][1, chamber_number, :, :, :]  # initialize

while True:
    ret, frame = vr.read()
    frame_number += 1
    # if mod(frame_number, 100) == 0:
    #     print(frame_number)

    # for fly in range(r['lines'].shape[2]):
    #     old = old_lines[fly, :, :]
    #     new = r['lines'][frame_number, chamber_number, fly, :, :]
    #     new, is_flipped, D = tk.fix_flips(old[0, :], new)
    #     lines[fly, :, :] = new
    # new_tails_fixed = lines[:, 0, :]
    # old_lines = lines.copy()
    # r['lines'][frame_number, chamber_number, :, :, :] = lines
    lines = np.clip(np.uint(r['lines'][frame_number, chamber_number, :, :, :]), 0, 10000)
    frame_with_tracks = fg.annotate(frame[80:, :, :] / 255,
                                    centers=lines[:,0,::-1],
                                    lines=lines)
    plt.cla()
    plt.imshow(frame_with_tracks)
    plt.pause(0.000001)
