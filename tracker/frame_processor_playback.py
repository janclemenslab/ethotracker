import numpy as np
import cv2

import tracker.ForeGround as fg
import tracker.Tracker as tk
from tracker.VideoReader import VideoReader
from tracker.BackGround import BackGround
from tracker.Results import Results

import matplotlib.pyplot as plt
plt.ion()


def init(vr, start_frame, threshold, nflies, file_name, num_bg_frames=100):
    res = Results()                     # init results object

    # A: estimate background
    bg = BackGround(vr)
    bg.estimate(num_bg_frames, start_frame)
    res.background = bg.background[:, :, 0]
    vr.reset()

    # B: detect chambers
    res.chambers = fg.get_chambers(res.background, chamber_threshold=1.0, min_size=20000, max_size=200000, kernel_size=17)
    # printf('found {0} chambers'.format( np.unique(res.chambers).shape[0]-1 ))
    # detect empty chambers
    vr.seek(start_frame)
    # 1. read frame and get foreground
    ret, frame = vr.read()
    foreground = fg.threshold(res.background - frame[:, :, 0], threshold * 255)
    # 2. segment and get flies and remove chamber if empty or "fly" too small
    labels = np.unique(res.chambers)
    area = np.array([fg.segment_center_of_mass(foreground * (res.chambers == label))[4] for label in labels])  # get fly size for each chamber
    labels[area < 20] = 0                                                  # mark empty chambers for deletion
    res.chambers, _, _ = fg.clean_labels(res.chambers, labels, force_cont=True)  # delete empty chambers
    # 3. get bounding boxes for non-empty chambers for cropping
    res.chambers_bounding_box = fg.get_bounding_box(res.chambers)  # get bounding boxes of remaining chambers

    # C: populate Results structure
    res.nflies = int(nflies)
    res.nchambers = int(np.max(res.chambers))
    res.file_name = file_name
    res.start_frame = int(start_frame)
    res.frame_count = int(start_frame)
    res.number_of_frames = int(vr.number_of_frames)

    res.centers = np.zeros((res.number_of_frames + 1000, res.nchambers, res.nflies, 2), dtype=np.float16)
    res.area = np.zeros((res.number_of_frames + 1000, res.nchambers, res.nflies), dtype=np.float16)
    res.lines = np.zeros((res.number_of_frames + 1000, res.nchambers, res.nflies, 2, 2), dtype=np.float16)
    res.led = np.zeros((res.number_of_frames + 1000, res.nflies, 1), dtype=np.float16)
    # save initialized results object
    res.status = "initialized"
    res.save(file_name=file_name[0:-4] + '.h5')
    # printf('saving init')
    print('found {0} fly bearing chambers'.format( res.nchambers ))
    return res


class Prc():
    def __init__(self, res):
        self.frame_processor = self._process_coroutine(res)

    def process(self, frame, res):
        next(self.frame_processor)
        res, foreground = self.frame_processor.send((frame, res))
        return res, foreground

    def _process_coroutine(self, res):
        """Coroutine for processing the frame.
        """

        # init data structures
        centers = np.zeros((res.nchambers, res.nflies, 2))
        old_centers = None

        lines = np.zeros((res.nchambers, res.nflies, 2, 2))
        old_lines = None

        area = np.zeros((1, res.nchambers, res.nflies))
        uni_chambers = np.unique(res.chambers).astype(np.int)
        chamber_slices = [None] * int(res.nchambers + 1)
        for ii in uni_chambers:
            chamber_slices[ii] = (np.s_[res.chambers_bounding_box[ii, 0, 0]:res.chambers_bounding_box[ii, 1, 0],
                                        res.chambers_bounding_box[ii, 0, 1]:res.chambers_bounding_box[ii, 1, 1]])
        # process
        while True:
            frame, res = yield
            res.frame_count = int(res.frame_count+1)
            foreground = fg.threshold(res.background - frame[:, :, 0], res.threshold * 255)
            # foreground = fg.erode(foreground.astype(np.uint8), kernel_size=4)
            # foreground = fg.dilate(foreground.astype(np.uint8), kernel_size=4)
            foreground = fg.close(foreground.astype(np.uint8), kernel_size=4)
            # import ipdb; ipdb.set_trace()
            # foreground = cv2.blur(foreground.astype(np.uint8)*255.0,(11,11))
            for ii in uni_chambers:
                if ii > 0:  # 0 is background
                    foreground_cropped = foreground[chamber_slices[ii]] * (res.chambers[chamber_slices[ii]] == ii)  # crop frame to current chamber
                    centers[ii - 1, :], labels, points, _, area[0, ii - 1] = fg.segment_center_of_mass(foreground_cropped)
                    # centers[ii-1, :, :], labels, points,  = fg.segment_cluster(foreground_cropped, num_clusters=res.nflies)

                    if points.shape[0] > 0:   # check that there we have not lost the fly in the current frame
                        for label in np.unique(labels):
                            lines[ii-1, label, :, :], _ = tk.fit_line(points[labels[:, 0] == label, :]) # need to make this more robust - based on median center and some pixels around that...

            if res.nflies > 1 and old_centers is not None and old_lines is not None:  # match centers across frames - not needed for one fly per chamber
                for ii in uni_chambers:
                    if ii > 0:
                        new_labels, centers[ii-1, :, :] = tk.match(old_centers[ii-1, :, :], centers[ii-1, :, :])
                        lines[ii-1, :, :, :] = lines[ii-1, new_labels, :, :]  # also re-order lines
            old_centers = np.copy(centers)  # remember

            if old_lines is not None:  # fix forward/backward flips
                for ii in uni_chambers:
                    if ii > 0:
                        if points.shape[0] > 0:   # check that we have not lost all flies in the current frame
                            for label in np.unique(labels):
                                lines[ii-1, label, :, :], is_flipped, D = tk.fix_flips(old_lines[ii-1, label, 0, :], lines[ii-1, label, :, :])
            old_lines = np.copy(lines)  # remember

            res.centers[res.frame_count, :, :, :] = centers
            res.lines[res.frame_count, :, 0:lines.shape[1], :, :] = lines
            res.area[res.frame_count, :] = 0
            yield res, foreground
