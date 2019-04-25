"""Manual annotation of videos."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from videoreader import VideoReader
import tracker.foreground as fg
import os
import h5py
import scipy
import skimage.segmentation
from scipy import ndimage


class LineDrawer(object):
    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x,y)
        ax.figure.canvas.draw()
        return (x, y)


# load video
root = '/Users/janc/'#'/Volumes/ukme04/#Common/chaining/dat'
recname = 'localhost-20180226_161811'#'localhost-20180316_144329'
filename = os.path.join(root, recname + '/' + recname + '.avi')
print(filename)
vr = VideoReader(filename)

# load tracker results
filename = os.path.join(root, recname + '/' + recname + '.h5')
with h5py.File(filename, 'r') as f:
    centers = f['centers'][:]
    bg = f['background'][:]


# fly identity matching
frame_number0 = 2500#2700
frame_offset = -122#-250# -122get_offset(frame, centers[:, chamber_number, :, :], frame_number, 250)
chamber_number = 0

pos = centers[:, chamber_number, :, :]
nflies = pos.shape[1]


ret, frame = vr.read(frame_number0 - frame_offset)
frame = frame[40:, :, :]

# detect when they're all in a bunch and annotate this frame
pos0 = pos[frame_number0, :, :]
print(pos0)
D = scipy.spatial.distance.pdist(pos0)
D = scipy.spatial.distance.squareform(D)
D_min = 30
print(D)
number_close_flies = np.sum(D < D_min, axis=0) - 1  # number of other flies current fly is close to (-1 to remove self-distance)
focal_fly = np.argmax(number_close_flies)
flybox_width = int(3.5 * D_min)
# find flies within box
good_flies = np.where(np.all(np.logical_and(pos0 - pos0[focal_fly,:] + flybox_width>0, pos0 - pos0[focal_fly,:] + flybox_width<2*flybox_width), axis=1))[0]

bg = bg[40:,:]  # crop bg like frame
bg_box = bg[int(pos0[focal_fly, 0]-flybox_width):int(pos0[focal_fly, 0]+flybox_width),
            int(pos0[focal_fly, 1]-flybox_width):int(pos0[focal_fly, 1]+flybox_width)]

plt.ion()
cm = plt.get_cmap('tab10', nflies).colors
for frame_number in range(frame_number0, frame_number0 + 400):
    ret, frame = vr.read()
    frame = frame[40:, :, :]
    fly_box = frame[int(pos0[focal_fly, 0]-flybox_width):int(pos0[focal_fly, 0]+flybox_width),
                    int(pos0[focal_fly, 1]-flybox_width):int(pos0[focal_fly, 1]+flybox_width), 0]
    dx = pos0[focal_fly, 1] - flybox_width
    dy = pos0[focal_fly, 0] - flybox_width

    f = 255 - (bg_box - fly_box)
    # f = fg.dilate(f.astype(np.uint8), 5)
    if frame_number == frame_number0:
        marker_positions = pos[frame_number, good_flies, :] - [dy, dx]
    else:
        marker_positions = centers[centers[:, 0] > 0, :]
    # marker_positions = np.vstack(([0, 0], marker_positions))
    marker_positions, *_ = fg.segment_cluster((-f+255)>0.35*255, marker_positions.shape[0])
    centers, _, _, _, _, ws = fg.segment_watershed(f, marker_positions, frame_dilation=7)

    plt.clf()
    plt.subplot(131)
    plt.imshow(f, cmap='gray')

    for cnt, fly in enumerate(good_flies):
        # current fly position
        plt.plot(pos[frame_number, fly, 1] - dx, pos[frame_number, fly, 0] - dy, '.', color=cm[fly])
        # fly trajectory
        plt.plot(pos[frame_number0:frame_number0+20, fly, 1] - dx, pos[frame_number0:frame_number0+20, fly, 0] - dy, '-', color=cm[fly], linewidth=0.5)
        plt.text(pos[frame_number, fly, 1] - dx + 1, pos[frame_number, fly, 0] - dy - 1, str(fly), color=cm[fly])

    plt.subplot(132)
    plt.imshow(ws, cmap='tab10')
    plt.plot(centers[:, 1], centers[:, 0], '.w')

    plt.subplot(133)
    plt.imshow(f, cmap='gray')
    plt.scatter(centers[:, 1], centers[:, 0], s=4, c=cm)

    plt.show()
    plt.pause(0.00001)

    # input('press key for next frame')  # wait for any key press
