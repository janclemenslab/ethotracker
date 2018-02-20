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
recname = 'localhost-20180213_132850'  # 'localhost-20180213_145725'
filename = os.path.join(root, recname + '/' + recname + '.avi')
print(filename)
vr = VideoReader(filename)

# load tracker results
filename = os.path.join(root, recname + '/' + recname + '.h5')
r = h5py.File(filename)

# vr.frame_rate = 40
frame_number = 0
frame_offset = 250
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

    frame_with_tracks = fg.annotate(frame[80:, :, :] / 255,
                                    centers=np.clip(np.uint(lines[:, 0, ::-1]), 0, 10000),
                                    lines=np.clip(np.uint(r['lines'][frame_number, chamber_number, :, :, :]), 0, 10000))
    plt.cla()
    plt.imshow(frame_with_tracks)
    plt.pause(0.000001)
