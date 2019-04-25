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
        line = plt.plot(x,y)
        ax.figure.canvas.draw()
        return (x, y)


def cropit(foreground, mask, bounding_box):
    chamber_slices = (np.s_[bounding_box[0, 0]:bounding_box[1, 0],
                            bounding_box[0, 1]:bounding_box[1, 1]])
    return foreground[chamber_slices] * mask[chamber_slices]  # crop frame to current chamber, chb+1 since 0 is background

def pushdata(old, new, axis=0):
    old = np.delete(old, 0, axis=axis)
    return np.append(old, new, axis=axis)

# load video
root = '/Volumes/ukme04/#Common/chaining/dat'
recname = 'localhost-20180226_161811'#'localhost-20180316_144329'
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
frame_number0 = 2900#2450#2700
frame_offset = -122#-250# -122get_offset(frame, centers[:, chamber_number, :, :], frame_number, 250)
chamber_number = 0
bounding_box = chamber_bounding_box[chamber_number+1]

pos = centers[:, chamber_number, :, :]
nflies = pos.shape[1]
cm = plt.get_cmap('tab10', nflies).colors

ret, frame = vr.read(frame_number0 - frame_offset)
frame = cropit(frame[:,:,0], chamber_mask, bounding_box)

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

bg = cropit(bg, chamber_mask, bounding_box)
bg_box = bg[int(pos0[focal_fly, 0]-flybox_width):int(pos0[focal_fly, 0]+flybox_width),
            int(pos0[focal_fly, 1]-flybox_width):int(pos0[focal_fly, 1]+flybox_width)]

# marker_positions = np.zeros((5, good_flies.shape[0],2))
# stack = np.zeros((bg_box.shape[0], bg_box.shape[1], 5), dtype=np.uint8)
# for frame_number in range(frame_number0, frame_number0 + 400):
#
#     for cnt, fn in enumerate(range(frame_number-5, frame_number)):
#         ret, ff = vr.read(fn)
#         ff = cropit(ff[:,:,0], chamber_mask, bounding_box)
#         fly_box = ff[int(pos0[focal_fly, 0]-flybox_width):int(pos0[focal_fly, 0]+flybox_width),
#                         int(pos0[focal_fly, 1]-flybox_width):int(pos0[focal_fly, 1]+flybox_width)]
#
#         stack[:,:,cnt] = 255 - (bg_box - fly_box)
#         # stack = pushdata(stack, 255 - (bg_box - fly_box), axis=2)
#         if frame_number == frame_number0:
#             marker_positions[cnt,:,:], *_ = fg.segment_cluster((-stack[:,:,cnt]+255)>0.35*255, marker_positions.shape[1])
#             if cnt>0:
#                 _, marker_positions[cnt, :, :] = tk.match(marker_positions[0, :, :], marker_positions[cnt, :, :])
#         else:
#             if cnt==good_flies.shape[0]-1:
#                 marker_positions = marker_positions[::-1,:,:]
#                 marker_positions[-1,:,:], *_ = fg.segment_cluster((-stack[:,:,cnt]+255)>0.35*255, marker_positions.shape[1])
#                 if cnt>0:
#                     _, marker_positions[-1,:,:] = tk.match(marker_positions[0, :, :], marker_positions[cnt, :, :])
#         # marker_positions[cnt,:,:], *_ = fg.segment_cluster((-stack[:,:,cnt]+255)>0.35*255, marker_positions.shape[1])
#         # if cnt>0:
#         #     _, marker_positions[cnt, :, :] = tk.match(marker_positions[0, :, :], marker_positions[cnt, :, :])
#
#
#     marker_positions = np.uintp(marker_positions)
#     markers = np.zeros(stack.shape, dtype=np.uint8)
#     for cnt, marker_position in enumerate(marker_positions):
#         print(marker_position)
#         markers[marker_position[:,0], marker_position[:,1], cnt] = np.arange(marker_position.shape[0])+1  # +1 since 0=background
#     # markers[marker_positions[2,:,0], marker_positions[2, :,1], cnt] = np.arange(marker_positions.shape[1])+1  # +1 since 0=background
#
#     # make segmentation more robust - dilation factor determines tolerance - bigger values allow catching the fly even if it moves quite a bit
#     frame_threshold=180
#     frame_dilation=3
#     marker_dilation=15
#     markers = fg.dilate(markers, marker_dilation)
#     bg_mask = stack < frame_threshold  # everything outside of mask is ignored - need to test robustness of threshold
#     stackd = fg.dilate(stack, frame_dilation)
#     labeled_frame = skimage.segmentation.watershed(stackd, markers, mask=bg_mask)
#     # labeled_frame = skimage.segmentation.watershed(stack, markers, mask=bg_mask)
#
#     # process segmentation
#     centers = np.zeros((5, good_flies.shape[0]+1, 2))
#     for cnt in range(5):
#         this_labeled_frame = labeled_frame[:,:,cnt]
#         labels = this_labeled_frame[this_labeled_frame > 0]  # get all foreground labels
#         points = fg.getpoints(this_labeled_frame > 0)  # get position of all foreground pixels
#         number_of_segments = np.unique(this_labeled_frame)[1:].shape[0]+1 # +1 since 0 is background
#         for ii in range(number_of_segments):  # [1:] since first unique values=0=background
#             centers[cnt, ii, :] = np.median(points[labels == ii, :], axis=0)
#
#     plt.ion()
#     plt.clf()
#     for cnt in range(5):
#         plt.subplot(2,5, cnt+1)
#         plt.imshow(stack[:,:,cnt], cmap='gray')
#         plt.scatter(centers[cnt,:,1], centers[cnt,:,0], c=cm, s=4)
#
#         plt.subplot(2,5, 5+ cnt+1)
#         plt.imshow(labeled_frame[:,:,cnt], cmap='tab10')
#         plt.scatter(centers[cnt,:,1], centers[cnt,:,0], c=cm, s=4)
#     plt.show()
#     plt.pause(0.0001)


plt.ion()
ret, frame = vr.read(frame_number0-1)
for frame_number in range(frame_number0, frame_number0 + 400):
    ret, frame = vr.read()

    frame = cropit(frame[:,:,0], chamber_mask, bounding_box)
    fly_box = frame[int(pos0[focal_fly, 0]-flybox_width):int(pos0[focal_fly, 0]+flybox_width),
                    int(pos0[focal_fly, 1]-flybox_width):int(pos0[focal_fly, 1]+flybox_width)]
    dx = pos0[focal_fly, 1] - flybox_width
    dy = pos0[focal_fly, 0] - flybox_width

    f = 255 - (bg_box - fly_box)
    # f = fg.dilate(f.astype(np.uint8), 5)
    if frame_number == frame_number0:
        marker_positions = pos[frame_number, good_flies, :] - [dy, dx]
    else:
        marker_positions = centers[centers[:, 0] > 0, :]

    marker_positions, *_ = fg.segment_cluster((-f+255)>0.35*255, marker_positions.shape[0])

    ws_mask = (-f+255)>0.35*255
    centers, _, _, _, _, ws = fg.segment_watershed(f, marker_positions, frame_threshold=180, frame_dilation=7, post_ws_mask=ws_mask)

    # import ipdb; ipdb.set_trace()
    # centers, *_ = fg.segment_connected_components(fg.erode(ws.astype(np.uint8),5))
    # print(centers)
    if frame_number == frame_number0:
        oldcenters = centers
    else:
         _, centers[1:,:] = tk.match(oldcenters[1:,:], centers[1:,:])
    #     print(oldcenters)
    #     print(centers)
    #     _, centers = tk.match(oldcenters, centers)
    oldcenters = centers
    # oldcenters = oldcenters*0.67 + 0.33*centers
    plt.clf()
    plt.subplot(121)
    plt.imshow(ws, cmap='tab10')
    plt.plot(centers[:, 1], centers[:, 0], '.w')

    plt.subplot(122)
    # plt.imshow(fly_box, cmap='gray')
    plt.imshow(f, cmap='gray')
    plt.scatter(centers[:, 1], centers[:, 0], s=4, c=cm)
    plt.scatter(oldcenters[:, 1], oldcenters[:, 0], s=2, c=cm)

    plt.show()
    plt.pause(0.00001)
    # import ipdb; ipdb.set_trace()

    # input('press key for next frame')  # wait for any key press
