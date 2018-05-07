"""Frame processor for tracking playback videos."""
import numpy as np
import logging

import tracker.ForeGround as fg
import tracker.Tracker as tk
from tracker.BackGround import BackGroundMax, BackGround
from tracker.attrdict import AttrDict
import matplotlib.pyplot as plt
plt.ion()


def init(vr, start_frame, threshold, nflies, file_name, num_bg_frames=100):
    """Prepare frame processor.

    Args:
        vr, start_frame, threshold, nflies, file_name, num_bg_frames=100
    Returns:
        res: initialized Results object
    """
    res = AttrDict()                     # init results object
    # A: estimate background
    res.frame_channel = 0  # red is best but hard to detect chamber!
    bg = BackGroundMax(vr)
    bg.estimate(num_bg_frames, start_frame)
    res.background = bg.background[:, :, res.frame_channel]

    # B: detect chambers
    # 0. detect chambers in background
    bg = BackGround(vr)  # use mean background since max background merged LED with last chamber
    bg.estimate(100, start_frame)
    res.chambers = fg.get_chambers(bg.background[:, :, res.frame_channel], chamber_threshold=1.0, min_size=35000, max_size=200000, kernel_size=17)
    # 1. read frame and get foreground
    foreground = fg.threshold(res.background - vr[0][:, :, res.frame_channel], threshold * 255)
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
    res.save(file_name[0:-4] + '.h5')
    # printf('saving init')
    logging.info(f'found {res.nchambers} fly bearing chambers')
    return res


class Prc():
    """Frame processor for tracking a frame.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.

    """

    def __init__(self, res):
        """Initialize and return a frame processor instance. Needs Results object."""
        self.frame_processor = self._process_coroutine(res)

    def process(self, frame, res):
        """Public interface for the frame processor. Hides the co-routine boiler plate code."""
        next(self.frame_processor)
        res, foreground = self.frame_processor.send((frame, res))
        return res, foreground

    def _process_coroutine(self, res):
        """Coroutine for processing the frame.

        Args:
            res: Results object
        Receives:
            frame:
        Yields:
            res: updated Results object
            foreground:
        """
        # init data structures
        centers = np.zeros((res.nchambers, res.nflies, 2))
        old_centers = None

        lines = np.zeros((res.nchambers, res.nflies, 2, 2))
        old_lines = None

        area = np.zeros((1, res.nchambers, res.nflies))

        # get list of chamber numbers and remove background "chamber"
        uni_chambers = np.unique(res.chambers).astype(np.int)
        uni_chambers = uni_chambers[uni_chambers > 0] - 1  # get rid of background (0) and -1 makes sure chamber1 ==unichamber 0

        # get slices for cropping foreground by chambers
        chamber_slices = [None] * int(res.nchambers)
        for chb in uni_chambers:
            # FIXME: chambers_bounding_box[chb+1...] since box[0] is full frame/background - maybe fix that by removing background? will probably break things in playback tracker
            chamber_slices[chb] = (np.s_[res.chambers_bounding_box[chb+1, 0, 0]:res.chambers_bounding_box[chb+1, 1, 0],
                                         res.chambers_bounding_box[chb+1, 0, 1]:res.chambers_bounding_box[chb+1, 1, 1]])

        # process
        while True:
            frame, res = yield  # get new frame
            res.frame_count = int(res.frame_count+1)
            foreground = fg.threshold(res.background - frame[:, :, res.frame_channel], res.threshold * 255)
            foreground = fg.erode(foreground.astype(np.uint8), kernel_size=4)  # get rid of artefacts from chamber border
            foreground = fg.close(foreground.astype(np.uint8), kernel_size=4)  # smooth out fly shapes
            # import ipdb; ipdb.set_trace()

            for chb in uni_chambers:
                foreground_cropped = foreground[chamber_slices[chb]] * (res.chambers[chamber_slices[chb]] == chb+1)  # crop frame to current chamber
                if res.nflies == 1:
                    centers[chb, :, :], labels, points, _, area[0, chb] = fg.segment_center_of_mass(foreground_cropped)
                else:
                    centers[chb, :, :], labels, points,  = fg.segment_cluster(foreground_cropped, num_clusters=res.nflies)

                if points.shape[0] > 0:   # check that we have not lost the fly in the current frame
                    for label in np.unique(labels):
                        lines[chb, label, :, :], _ = tk.fit_line(points[labels[:, 0] == label, :])  # need to make this more robust - based on median center and some pixels around that...

                if res.nflies > 1 and old_centers is not None and old_lines is not None:  # match centers across frames - not needed for one fly per chamber
                    # for chb in uni_chambers:
                    new_labels, centers[chb, :, :] = tk.match(old_centers[chb, :, :], centers[chb, :, :])
                    lines[chb, :, :, :] = lines[chb, new_labels, :, :]  # also re-order lines
            old_centers = np.copy(centers)  # remember

            if old_lines is not None:  # fix forward/backward flips
                for chb in uni_chambers:
                    if points.shape[0] > 0:   # check that we have not lost all flies in the current frame
                        for label in np.unique(labels):
                            lines[chb, label, :, :], is_flipped, D = tk.fix_flips(old_lines[chb, label, 0, :], lines[chb, label, :, :])
            old_lines = np.copy(lines)  # remember

            res.centers[res.frame_count, :, :, :] = centers
            res.lines[res.frame_count, :, 0:lines.shape[1], :, :] = lines
            res.area[res.frame_count, :] = 0
            yield res, foreground
