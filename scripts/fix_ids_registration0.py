"""Manual annotation of videos."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from videoreader import VideoReader
import tracker.foreground as fg
import tracker.tracker as tk

import os
import h5py
import scipy
import skimage
# import skimage.segmentation
from scipy import ndimage
import cv2

class LineDrawer(object):
    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x, y)
        ax.figure.canvas.draw()
        return (x, y)


def cropit(foreground, mask, bounding_box):
    chamber_slices = (np.s_[bounding_box[0, 0]:bounding_box[1, 0],
                            bounding_box[0, 1]:bounding_box[1, 1]])
    return foreground[chamber_slices] * mask[chamber_slices]  # crop frame to current chamber, chb+1 since 0 is background


def pushdata(old, new, axis=0):
    old = np.delete(old, 0, axis=axis)
    return np.append(old, new, axis=axis)


def register(crop0, crop1, old_center, warp_mode=cv2.MOTION_EUCLIDEAN, number_of_iterations=100, termination_eps=1e-12):
    warp_matrix = np.eye(2, 3, dtype=np.float32)  # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)  # Define termination criteria

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(crop0, crop1, warp_matrix, warp_mode, criteria)
    crop0_aligned = cv2.warpAffine(crop1, warp_matrix, crop0.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    new_center = np.dot(warp_matrix, np.array([old_center[0], old_center[1], 1]))
    return new_center


# load video
root = '/Volumes/ukme04/#Common/chaining/dat'  # '/Users/janc/'#
recname = 'localhost-20180222_130351'#'localhost-20180226_161811'  # 'localhost-20180316_144329'
filename = os.path.join(root, recname + '/' + recname + '.avi')
print(filename)
vr = VideoReader(filename)

# load tracker results
filename = os.path.join(root, recname + '/' + recname + '.h5')
with h5py.File(filename, 'r') as f:
    centers = f['centers'][:]
    bg = f['background'][:]
    chamber_mask = f['chambers'][:]
    chamber_bounding_box = f['chambers_bounding_box'][:]

# fly identity matching
frame_number0 = 66350#2400  # 2500#2450#2700
frame_offset = 0#-122  # -250# -122get_offset(frame, centers[:, chamber_number, :, :], frame_number, 250)
chamber_number = 0
bounding_box = chamber_bounding_box[chamber_number+1]

pos = centers[:, chamber_number, :, :]
nflies = pos.shape[1]
cm = plt.get_cmap('tab10', nflies).colors

ret, frame = vr.read(frame_number0 - frame_offset)
frame = cropit(frame[:, :, 0], chamber_mask, bounding_box)

# detect when they're all in a bunch and annotate this frame
pos0 = pos[frame_number0, :, :]
print(pos0)
D = scipy.spatial.distance.pdist(pos0)
D = scipy.spatial.distance.squareform(D)
D_min = 30
print(D)
number_close_flies = np.sum(D < D_min, axis=0) - 1  # number of other flies current fly is close to (-1 to remove self-distance)
focal_fly = np.argmax(number_close_flies)
flybox_width = int(3.9 * D_min)
# find flies within box
good_flies = np.where(np.all(np.logical_and(pos0 - pos0[focal_fly, :] + flybox_width > 0, pos0 - pos0[focal_fly, :] + flybox_width < 2*flybox_width), axis=1))[0]

bg = cropit(bg, chamber_mask, bounding_box)
bg_box = bg[int(pos0[focal_fly, 0]-flybox_width):int(pos0[focal_fly, 0]+flybox_width),
            int(pos0[focal_fly, 1]-flybox_width):int(pos0[focal_fly, 1]+flybox_width)]

# Define the motion model


def overlay(im1, im2):
    sz = im1.shape
    im_aligned = np.zeros((sz[0], sz[1], 3), dtype=np.uint8)
    im_aligned[:, :, 0] = im1
    im_aligned[:, :, 1] = im2
    im_aligned[:, :, 2] = im1/2 + im2/2
    return im_aligned


dx = pos0[focal_fly, 1] - flybox_width
dy = pos0[focal_fly, 0] - flybox_width
marker_positions = pos[frame_number0, good_flies, :] - [dy, dx]

ret, frame = vr.read(frame_number0-1)
frame = cropit(frame[:, :, 0], chamber_mask, bounding_box)
fly_box = frame[int(pos0[focal_fly, 0]-flybox_width):int(pos0[focal_fly, 0]+flybox_width),
                int(pos0[focal_fly, 1]-flybox_width):int(pos0[focal_fly, 1]+flybox_width)]

f = 255 - (bg_box - fly_box)


ret, frame1 = vr.read(frame_number0)
frame1 = cropit(frame1[:, :, 0], chamber_mask, bounding_box)
fly_box = frame1[int(pos0[focal_fly, 0]-flybox_width):int(pos0[focal_fly, 0]+flybox_width),
                 int(pos0[focal_fly, 1]-flybox_width):int(pos0[focal_fly, 1]+flybox_width)]
f1 = 255 - (bg_box - fly_box)

bw = 15

plt.ion()
plt.gcf().set_size_inches(20, 10)
plt.subplot(1, 2, 1)
plt.imshow(f)
plt.plot(marker_positions[:,1],marker_positions[:,0], '.r')
plt.subplot(1, 2, 2)
plt.imshow(f1)
plt.plot(marker_positions[:,1],marker_positions[:,0], '.r')

# ARGS
for fly in range(marker_positions.shape[0]):
    crop0 = f[int(marker_positions[fly, 0]-bw):int(marker_positions[fly, 0]+bw), int(marker_positions[fly, 1]-bw):int(marker_positions[fly, 1]+bw)].astype(np.uint8)
    crop1 = f1[int(marker_positions[fly, 0]-bw):int(marker_positions[fly, 0]+bw), int(marker_positions[fly, 1]-bw):int(marker_positions[fly, 1]+bw)].astype(np.uint8)
    nc = register(crop0, crop1, [bw, bw])
    new_center = marker_positions[fly,::-1] + nc.T - [bw,bw]
    plt.plot(new_center[0], new_center[1], '.g')
