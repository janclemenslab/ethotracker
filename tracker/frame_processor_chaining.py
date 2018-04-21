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
        """Process."""
        # init
        centers = np.zeros((res.nchambers, res.nflies, 2))
        old_centers = None

        lines = np.zeros((res.nchambers, res.nflies, 2, 2))
        old_lines = None

        frame_error = np.zeros((res.nchambers, 1), dtype=np.uint8)

        area = np.zeros((1, res.nchambers, res.nflies))

        # get list of chamber numbers and remove background "chamber"
        uni_chambers = np.unique(res.chambers).astype(np.int)
        uni_chambers = uni_chambers[uni_chambers > 0] - 1  # get rid of background (0) and -1 makes sure chamber1 ==unichamber 0

        # get slices for cropping foreground by chambers
        chamber_slices = [None] * int(res.nchambers)
        for ii in uni_chambers:
            # FIXME: chambers_bounding_box[ii+1...] since box[0] is full frame/background - maybe fix that by removing background? will probably break things in playback tracker
            chamber_slices[ii] = (np.s_[res.chambers_bounding_box[ii+1, 0, 0]:res.chambers_bounding_box[ii+1, 1, 0],
                                        res.chambers_bounding_box[ii+1, 0, 1]:res.chambers_bounding_box[ii+1, 1, 1]])
        # process
        while True:
            frame, res = yield
            res.frame_count = int(res.frame_count+1)
            foreground0 = res.background - frame[:, :, 0]
            foreground = fg.threshold(res.background - frame[:, :, 0], res.threshold * 255)
            foreground = fg.erode(foreground, kernel_size=4)
            foreground = cv2.medianBlur(foreground, 3)  # get rid of specks
            for chb in uni_chambers:
                FRAME_PROCESSING_ERROR = False  # flag if there were errors during processing of this frames so we can fall back to segment_cluster
                foreground_cropped = foreground[chamber_slices[chb]] * (res.chambers[chamber_slices[chb]] == chb+1)  # crop frame to current chamber, chb+1 since 0 is background
                if old_centers is None:  # on first pass get initial values - this only works if first frame produces the correct segmentation
                    centers[chb, :, :], labels, points,  = fg.segment_cluster(foreground_cropped, num_clusters=res.nflies)
                    print(f"{res.frame_count}: restarting - clustering")
                    frame_error[chb] = 1
                else:  # for subsequent frames use connected components and split those with multiple flies
                    this_centers, this_labels, points, _, this_size, labeled_frame = fg.segment_connected_components(
                                                                                            foreground_cropped, minimal_size=5)
                    # assigning flies to conn comps
                    fly_conncomps, flycnt, flybins, cnt = fg.find_flies_in_conn_comps(labeled_frame, old_centers[chb, :, :], max_repeats=5, initial_dilation_factor=10, repeat_dilation_factor=5)
                    # if all flies are assigned a conn comp and all conn comps contain a fly - proceed
                    # alternatively, we could simply "delete" empty conn comps
                    if flycnt[0] == 0 and not np.any(flycnt[1:] == 0):
                        flybins = flybins[1:] - 1  # remove bg bin and shift so it starts at 0 for indexing

                        this_foreground0 = 255-foreground0[chamber_slices[chb]]  # non-thresholded foreground for watershed
                        labels = this_labels.copy()  # copy for new labels
                        # split conn comps with multiple flies using clustering
                        for con in np.uintp(flybins[flycnt > 1]):
                            # 1. get bounding box around current comp for cropping frame
                            con_frame = this_foreground0.copy()
                            con_frame[fg.erode(labeled_frame != con, 10) == 1] = 100  # mark background - erode to add padding around flies
                            con_bb = fg.get_bounding_box(fg.dilate(con_frame != 100, 15) == 1)  # dilate to get enough margin around conn comp
                            con_bb = con_bb[1][:, ::-1]  # [0] is bg, ::-1 order x,y
                            con_offset = np.min(con_bb, axis=0)  # get upper left corner of box - needed to transform positions back into global chamber coords

                            # 2. crop around current comp
                            con_frame = this_foreground0.copy()
                            con_frame = fg.crop(con_frame, np.ravel(con_bb))
                            # mask indicating other comps -  we want to ignore those in the current comp
                            con_frame_mask = fg.erode(fg.crop(labeled_frame, np.ravel(con_bb)) != con, 10) == 1

                            # 3. segment using clustering
                            #  - if >2 flies then use cluster results as seeds for watershed - usually refines fly positions if flies in a bunch
                            con_frame_thres = (-con_frame+255) > res.threshold*255  # threshold current patch
                            con_frame_thres_erode = fg.erode(con_frame_thres, 7)  # amplify gaps/sepration between flies
                            con_frame_thres_erode[con_frame_mask] = 0  # mask out adjecent flies/patches
                            # 4. if 2 flies in comp we stop here,
                            if flycnt[con] == 1:
                                con_centers, con_labels, con_points = fg.segment_cluster(con_frame_thres_erode, num_clusters=flycnt[con])
                            # 4. if >2 flies in conn comp we watershed
                            else:  # only use additional watershed step when >2 flies in conn comp
                                # 4a. try to set threshold as high as possible - we know the upper bound for the size of a fly, and we know how many flies there are in the current con comp...
                                thres = res.threshold  # initialize with standard thres
                                fly_area = 150
                                while np.sum(con_frame_thres) > flycnt[con]*fly_area:  # shouldn't we use a con_frame_thres with outside flies masked out???
                                    thres += 0.01  # increment thres as long as there are too many foreground pixels for the number of flies
                                    con_frame_thres = (-con_frame+255) > thres*255
                                con_frame_thres = fg.erode(con_frame_thres, 3)
                                # 4a. locally adaptive threshold finds gaps between flies
                                ada = cv2.adaptiveThreshold((-con_frame+255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1)
                                con_frame_thres[ada == 0] = 0

                                # mask out adjacent flies/patches
                                con_frame[con_frame_mask] = 0

                                marker_positions = np.vstack(([0, 0], old_centers[chb, np.where(fly_conncomps==con), :][0] - con_offset[::-1]))# use old positions instead
                                con_centers, con_labels, con_points, _, _, ll = fg.segment_watershed(con_frame, marker_positions, frame_threshold=180, frame_dilation=7, post_ws_mask=con_frame_thres)

                                # plt.subplot(121)
                                # plt.cla()
                                # plt.imshow(con_frame)
                                # plt.plot(marker_positions[:,1], marker_positions[:,0], '.w')
                                # plt.subplot(122)
                                # plt.imshow(ll)
                                # plt.show()
                                # plt.pause(0.00001)

                                con_labels = con_labels - 2  # labels starts at 1 - "1" is background and we want it to start at "0" for use as index
                                # only keep foreground points/labels
                                con_points = con_points[con_labels[:, 0] >= 0, :]
                                con_labels = con_labels[con_labels[:, 0] >= 0, :]
                            con_points = con_points + con_offset[::-1]  # current point coordinates are in cropped frame - transform to chamber coords
                            # input('hit')
                            # delete old labels and points - if we erode/dilate con comp we will have fewer/more points
                            try:
                                points = points[labels[:, 0] != con, :]
                                labels = labels[labels[:, 0] != con]
                            except Exception as e:
                                print(e)
                                FRAME_PROCESSING_ERROR = True
                                frame_error[chb] = 4
                            # append new labels and points
                            if labels.shape[0] == 0:  # if all flies in a single component then labels/points will be empty we use default since max op will error
                                new_con_labels = 100 + con_labels
                            else:
                                new_con_labels = np.max(labels) + 10 + con_labels
                            labels = np.append(labels, new_con_labels, axis=0)
                            points = np.append(points, con_points, axis=0)

                        # make labels consecutive numbers again so we can use them as indices
                        labels, _, _ = fg.clean_labels(labels)
                        # calculate positions for all flies
                        try:
                            centers[chb, :, :] = [np.median(points[labels[:, 0] == label, :], axis=0) for label in np.unique(labels)]
                        except Exception as e:
                            print(e)
                            FRAME_PROCESSING_ERROR = True
                            frame_error[chb] = 5

                        if FRAME_PROCESSING_ERROR:
                            print(f"{res.frame_count}: we lost at least one fly (or something else) - falling back to segment cluster - should mark frame as potential jump")
                            centers[chb, :, :], labels, points,  = fg.segment_cluster(foreground_cropped, num_clusters=res.nflies)

                    else:  # if still flies w/o conn compp fall back to segment_cluster
                        print(f"{res.frame_count}: {flycnt[0]} outside of the conn comps or conn comp {np.where(flycnt[1:] == 0)} is empty - falling back to segment cluster - should mark frame as potential jump")
                        centers[chb, :, :], labels, points,  = fg.segment_cluster(foreground_cropped, num_clusters=res.nflies)
                        frame_error[chb] = 3

                    if points.shape[0] > 0:   # make sure we have not lost the flies in the current frame
                        for label in np.unique(labels):
                            lines[chb, label, :, :], _ = tk.fit_line(points[labels[:, 0] == label, :])  # need to make this more robust - based on median center and some pixels around that...

            if res.nflies > 1 and old_centers is not None and old_lines is not None:  # match centers across frames - not needed for one fly per chamber
                for ii in uni_chambers:
                    new_labels, centers[chb, :, :] = tk.match(old_centers[chb, :, :], centers[chb, :, :])
                    lines[chb, :, :, :] = lines[chb, new_labels, :, :]  # also re-order lines
            old_centers = np.copy(centers)  # remember

            if old_lines is not None:  # fix forward/backward flips
                for ii in uni_chambers:
                    if points.shape[0] > 0:   # check that we have not lost all flies in the current frame
                        for label in np.unique(labels):
                            lines[chb, label, :, :], is_flipped, D = tk.fix_flips(old_lines[chb, label, 0, :], lines[chb, label, :, :])
            old_lines = np.copy(lines)  # remember

            res.centers[res.frame_count, :, :, :] = centers
            res.lines[res.frame_count, :, 0:lines.shape[1], :, :] = lines
            res.frame_error[res.frame_count, :, :] = frame_error
            res.area[res.frame_count, :] = 0
            yield res, foreground
