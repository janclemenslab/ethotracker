import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
# import cv2
from tracker.VideoReader import VideoReader
import tracker.ForeGround as fg
# import tracker.Tracker as tk
import os
import h5py

class LineDrawer(object):
    # lines = []
    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x,y)
        ax.figure.canvas.draw()
        return (x, y)

# load video
root = '/Volumes/ukme04/#Common/chaining/dat'
recname = 'localhost-20180226_161811'
filename = os.path.join(root, recname + '/' + recname + '.avi')
print(filename)
vr = VideoReader(filename)

# load tracker results
filename = os.path.join(root, recname + '/' + recname + '.h5')
with h5py.File(filename, 'r') as f:
    lines = f['lines'][:]
    centers = f['centers'][:]

# read first frame to determine offset
chamber_number = 0
frame_number = 2000
ret, frame = vr.read(frame_number)
frame = frame[40:, :, :]
frame_offset = -250  

# plot initial estimate of offset
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
pos = np.clip(np.uint(centers[frame_number + frame_offset, chamber_number, :,:]), 0, 10000)
frame_with_tracks = fg.annotate(frame / 255, centers=pos)
plt.imshow(frame_with_tracks)
ax0 = plt.gca()

# draw slider
axcolor = 'lightgoldenrodyellow'
axoffset = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
soffset = Slider(axoffset, 'offset', -500, 0, valinit=-frame_offset)

# slider callback
def update(val):
    frame_offset = soffset.val
    pos = np.clip(np.uint(centers[frame_number + frame_offset, chamber_number, :,:]), 0, 10000)
    frame_with_tracks = fg.annotate(frame / 255, centers=pos)
    ax0.imshow(frame_with_tracks)
    fig.canvas.draw_idle()
soffset.on_changed(update)
plt.show()

# on close figure - save slider value
frame_offset = int(soffset.val)
print(f"offset: {frame_offset}")

# now annotate flies
frame_numbers = np.uintp(list(range(2000, vr.number_of_frames, 5000)))
print(frame_numbers.shape)
nflies = centers.shape[2]
flybox_width = 30
flybox = []
flyline = []
flyid = []
flyframe = []
ld = LineDrawer()
plt.ion()
plt.gcf().set_size_inches(4, 4)
for frame_number in frame_numbers:
    ret, frame = vr.read(frame_number)
    frame = frame[40:, :, :]

    for fly0 in range(nflies):
        pos = centers[frame_number + frame_offset, chamber_number, fly0,:]
        flybox.append(frame[pos[0]-flybox_width:pos[0]+flybox_width, pos[1]-flybox_width:pos[1]+flybox_width,0])
        flyid.append(fly0)
        flyframe.append(frame_number + frame_offset)
        plt.cla()
        plt.imshow(flybox[-1], cmap='gray')
        plt.title(frame_number)
        # plt.clim([0, 255])
        flyline.append(ld.draw_line())

    print(flyid)
    print(flyline)

# save results
with h5py.File(f"{recname}_flybox.h5") as f:
    f.create_dataset('flybox', data=flybox, compression=gzip)
    f.create_dataset('flyline', data=flyline, compression=gzip)
    f.create_dataset('flyframe', data=flyframe, compression=gzip)
    f.create_dataset('flyid', data=flyid, compression=gzip)
