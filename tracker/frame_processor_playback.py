import numpy as np
import cv2

import tracker.ForeGround as fg
import tracker.Tracker as tk

import matplotlib.pyplot as plt
plt.ion()


class Prc():
    def __init__(self, res):
        self.frame_processor = self.process(res)

    def prc(self, frame, res):
        next(self.frame_processor)
        res, foreground = self.frame_processor.send((frame, res))
        return res, foreground

    def process(self, res):
        # init

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
                    # centers[ii - 1, :], labels, points, _, area[0, ii - 1] = fg.segment_center_of_mass(foreground_cropped)
                    centers[ii-1, :, :], labels, points,  = fg.segment_cluster(foreground_cropped, num_clusters=res.nflies)

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
