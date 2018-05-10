"""Manual annotation of videos"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from videoreader import VideoReader
import tracker.foreground as fg
import os
import h5py
import scipy


class LineDrawer(object):
    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x,y)
        ax.figure.canvas.draw()
        return (x, y)


def get_offset(frame, centers, frame_number: int, frame_offset: int =-250) -> int:
    """Determine offset between VideoReader and tracks."""
    # plot initial estimate of offset
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    pos = np.clip(np.uint(centers[frame_number + frame_offset, :,:]), 0, 10000)
    frame_with_tracks = fg.annotate(frame / 255, centers=pos)
    plt.imshow(frame_with_tracks)
    ax0 = plt.gca()

    # draw slider
    axcolor = 'lightgoldenrodyellow'
    axoffset = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    soffset = Slider(axoffset, 'offset', -500, 0, valinit=-frame_offset)  # QUESTION: Should `valinit` be positive?

    # slider callback
    def update(val):
        frame_offset = int(soffset.val)
        pos = np.clip(np.uint(centers[frame_number + frame_offset, :,:]), 0, 10000)
        frame_with_tracks = fg.annotate(frame / 255, centers=pos)
        ax0.imshow(frame_with_tracks)
        fig.canvas.draw_idle()
    soffset.on_changed(update)
    plt.show()
    # on close figure - save slider value
    return int(soffset.val)


def annotate_orientations(vr, frame_numbers, centers, frame_offset, flybox_width=30):
    """Annotate fly orientation."""
    nflies = centers.shape[1]
    fly = {'box': [], 'line': [], 'id': [], 'frame': []}
    ld = LineDrawer()
    plt.ion()
    plt.gcf().set_size_inches(4, 4)
    for frame_cnt, frame_number in enumerate(frame_numbers):
        print(f"   Loading frame {frame_number} ({frame_cnt}/{len(frame_numbers)}).")
        ret, frame = vr.read(frame_number)
        frame = frame[40:, :, :]

        for fly0 in range(nflies):
            pos = centers[int(frame_number) + frame_offset, fly0, :]
            if np.max(pos)==0:
                print('all pos zeros - skipping that fly/frame')
                continue
            fly['box'].append(frame[int(pos[0]-flybox_width):int(pos[0]+flybox_width), int(pos[1]-flybox_width):int(pos[1]+flybox_width), 0])
            fly['id'].append(fly0)

            fly['frame'].append(frame_number + frame_offset)
            # mark center
            plt.cla()
            plt.imshow(fly['box'][-1], cmap='gray')
            plt.plot(flybox_width, flybox_width, '.g')
            plt.title(frame_number)
            # plt.clim([0, 255])
            fly['line'].append(ld.draw_line())
    return fly


def annotate_wingextension(vr, frame_numbers, centers, frame_offset, flybox_width=30):
    # wing = 0 - nothing, 1 - left extended, 2 - right extended, -1 - weird
    # maybe also add angle??
    nflies = centers.shape[1]
    fly = {'box': [], 'wing': [], 'id': [], 'frame': []}
    plt.ion()
    plt.gcf().set_size_inches(4, 4)
    for frame_cnt, frame_number in enumerate(frame_numbers):
        print(f"   Loading frame {frame_number} ({frame_cnt}/{len(frame_numbers)}).")
        ret, frame = vr.read(frame_number)
        frame = frame[40:, :, :]

        for fly0 in range(nflies):
            pos = centers[int(frame_number) + frame_offset, fly0, :]
            if np.max(pos) == 0:
                print('all pos zeros - skipping that fly/frame')
                continue
            fly['box'].append(frame[int(pos[0]-flybox_width):int(pos[0]+flybox_width), int(pos[1]-flybox_width):int(pos[1]+flybox_width), 0])
            fly['id'].append(fly0)

            fly['frame'].append(frame_number + frame_offset)
            # mark center
            plt.cla()
            plt.imshow(fly['box'][-1], cmap='gray')
            plt.plot(flybox_width, flybox_width, '.g')
            plt.title(frame_number)
            # plt.clim([0, 255])
            results = input('wing state? [1 - no ext, 2 - left ext, 3 right ext, 4 - weird1]: ')
            fly['wing'].append(results)
    return fly


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
    chbb = f['chambers_bounding_box'][:]

# read first frame to determine offset
nflies = centers.shape[2]
chamber_number = 0
frame_number = 2000
ret, frame = vr.read(frame_number)
# frame = frame[40:, :, :]

frame = fg.crop(frame, np.ravel(chbb[chamber_number+1][:, ::-1]))
frame_offset = 0#get_offset(frame, centers[:, chamber_number, :, :], frame_number, 250)
print(f"offset: {frame_offset}")

# annotate fly orientation
frame_numbers = np.uintp(list(range(2000, vr.number_of_frames, 10000)))
plt.ion()
fly = annotate_orientations(vr, frame_numbers, centers[:, chamber_number, :, :], frame_offset, flybox_width=30)
#
# # annotate fly wing startTracking
# # fly = annotate_wingextension(vr, frame_numbers, centers[:, chamber_number, :, :], frame_offset, flybox_width=30)
#
# # fly identity matching
# # check_identities(vr, frame_numbers, centers, frame_offset, flybox_width=30)
# flybox_width = 40
# tails = lines[:,chamber_number,:,0,:]
# heads = lines[:,chamber_number,:,1,:]
#
# plt.ion()
# plt.gcf().set_size_inches(40, 10)
# plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
#
# for frame_number in frame_numbers:
#     ret, frame = vr.read(frame_number)
#     frame = fg.crop(frame, np.ravel(chbb[chamber_number+1][:, ::-1]))
#     T = int(frame_number + frame_offset)
#     for fly0 in range(nflies):
#         pos = centers[T, chamber_number, fly0,:]
#         flybox = frame[np.uintp(pos[0]-flybox_width):np.uintp(pos[0]+flybox_width), np.uintp(pos[1]-flybox_width):np.uintp(pos[1]+flybox_width),0]
#         # rotate flies - all flies should look up
#         flyangle = np.degrees(np.arctan2(tails[T,fly0,0]-heads[T,fly0,0], tails[T,fly0,1]-heads[T,fly0,1]))
#         flybox_rot = scipy.ndimage.rotate(flybox, -flyangle, reshape=False)
#         plt.subplot(2,nflies+1, fly0+1)
#         plt.imshow(flybox_rot[10:-10,10:-10], cmap='gray')
#         plt.clim([0, 255])
#         # original fly frame with annotation
#         plt.subplot(2,nflies+1, nflies+fly0+2)
#         plt.cla()
#         plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
#         plt.imshow(flybox, cmap='gray')
#         plt.plot(lines[T,chamber_number,fly0,:,0].T-pos[1]+flybox_width, lines[T,chamber_number,fly0,:,1].T-pos[0]+flybox_width, 'b')
#         # plt.plot(tails[T,fly0,0]-pos[1]+flybox_width, tails[T,fly0,1]-pos[0]+flybox_width, '.g')
#         # plt.plot(heads[T,fly0,0]-pos[1]+flybox_width, heads[T,fly0,1]-pos[0]+flybox_width, '.r')
#         plt.clim([0, 255])
#         plt.pause(0.0001)
