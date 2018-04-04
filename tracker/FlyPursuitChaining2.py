"""Track chaining videos."""
import numpy as np
import cv2
import argparse
import time
import sys
import traceback
import os
import yaml

from .VideoReader import VideoReader
from .BackGround import BackGroundMax
from .Results import Results
import tracker.ForeGround as fg
import tracker.Tracker as tk

import matplotlib.pyplot as plt
plt.ion()


def init(vr, start_frame, threshold, nflies, file_name, num_bg_frames=1000, annotationfilename=None):
    # TODO:
    #  refactor - tracker needs: background, chamber mask, chmaber bounding box (for slicing)
    #  provide these as args, if no chamber mask and box use full frame
    #  chambers are detected automatically - e.g. in playback - or manually annotated
    #  should be: binary image for mask and rect corrds for box (or infer box from mask)
    #  background should be: matrix

    res = Results()                     # init results object
    bg = BackGroundMax(vr)
    bg.estimate(num_bg_frames, start_frame)
    res.background = bg.background[:, :, 0]
    vr.reset()

    # load annotation filename
    annotationfilename = os.path.splitext(file_name)[0] + '_annotated.txt'
    with open(annotationfilename, 'r') as stream:
        a = yaml.load(stream)

    # LED mask
    res.led_mask = np.zeros((vr.frame_width, vr.frame_height), dtype=np.uint8)
    res.led_mask = cv2.circle(res.led_mask, (int(a['rectCenterY']), int(a['rectCenterX'])), int(a['rectRadius']), color=[1, 1, 1], thickness=-1)
    res.led_coords = fg.get_bounding_box(res.led_mask)  # get bounding boxes of remaining chambers
    # chambers mask
    res.chambers = np.zeros((vr.frame_width, vr.frame_height), dtype=np.uint8)
    res.chambers = cv2.circle(res.chambers, (int(a['centerY']), int(a['centerX'])), int(a['radius']+10), color=[1, 1, 1], thickness=-1)

    # chambers bounding box
    res.chambers_bounding_box = fg.get_bounding_box(res.chambers)  # get bounding boxes of remaining chambers
    # FIXME: no need to pre-pend background anymore
    res.chambers_bounding_box[0] = [[0, 0], [res.background.shape[0], res.background.shape[1]]]

    # init Results structure
    res.nflies = int(a['nFlies'])
    res.nchambers = int(np.max(res.chambers))
    res.file_name = file_name
    res.start_frame = int(start_frame)
    res.frame_count = int(start_frame)
    res.number_of_frames = int(vr.number_of_frames)
    if res.number_of_frames <= 0:
        print('neg. number of frames detected - fallback to 3h recording')
        res.number_of_frames = vr.frame_rate * 60 * 60 * 3  # fps*sec*min*hour

    res.centers = np.zeros((res.number_of_frames + 1000, res.nchambers, res.nflies, 2), dtype=np.float16)
    res.area = np.zeros((res.number_of_frames + 1000, res.nchambers, res.nflies), dtype=np.float16)
    res.lines = np.zeros((res.number_of_frames + 1000, res.nchambers, res.nflies, 2, 2), dtype=np.float16)
    res.led = np.zeros((res.number_of_frames + 1000, res.nflies, 1), dtype=np.float16)
    res.quality = np.zeros((res.number_of_frames + 1000, res.nchambers, res.nflies, 2, 2), dtype=np.float16)
    res.frame_error = np.zeros((res.number_of_frames + 1000, res.nchambers, 1), dtype=np.uint8)
    res.frame_error_codes = None  # should be dictionary mapping error codes to messages
    # save initialized results object
    res.status = "initialized"
    res.save(file_name=file_name[0:-4] + '.h5')
    print(f'found {res.nchambers} fly bearing chambers')
    return res


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
            foreground = fg.erode(foreground.astype(np.uint8), kernel_size=4)
            foreground = cv2.medianBlur(foreground.astype(np.uint8), 3)  # get rid of specks
            for chb in uni_chambers:
                foreground_cropped = foreground[chamber_slices[chb]] * (res.chambers[chamber_slices[chb]] == chb+1)  # crop frame to current chamber, chb+1 since 0 is background
                if old_centers is None:  # on first pass get initial values - this only works if first frame produces the correct segmentation
                    centers[chb, :, :], labels, points,  = fg.segment_cluster(foreground_cropped, num_clusters=res.nflies)
                    frame_error[chb] = 1
                else:  # for subsequent frames use connected components and split those with multiple flies
                    this_centers, this_labels, points, _, this_size, labeled_frame = fg.segment_connected_components(
                                                                                            foreground_cropped, minimal_size=5)
                    # assigning flies to conn comps
                    # FIXME: make flybins proper indices: np.uintp(flybins + 0.5)
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
                            con_frame[fg.erode(np.uint8(labeled_frame != con), 10) == 1] = 100  # mark background - erode to add padding around flies
                            con_bb = fg.get_bounding_box(fg.dilate(np.uint8(con_frame != 100), 15) == 1)  # dilate to get enough margin around conn comp
                            con_bb = con_bb[1][:, ::-1]  # [0] is bg, ::-1 order x,y
                            con_offset = np.min(con_bb, axis=0)  # get upper left corner of box - needed to transform positions back into global chamber coords

                            # 2. crop around current comp
                            con_frame = this_foreground0.copy()
                            con_frame = fg.crop(con_frame, np.ravel(con_bb))
                            # mask indicating other comps -  we want to ignore those in the current comp
                            con_frame_mask = fg.erode(np.uint8(fg.crop(labeled_frame, np.ravel(con_bb)) != con), 10) == 1

                            # 3. segment using clustering
                            #  - if 2 flies in comp we stop here,
                            #  - if >2 flies then use cluster results as seeds for watershed - usually refines fly positions if flies in a bunch
                            con_frame_thres = (-con_frame + 255) > res.threshold*255  # threshold current patch
                            con_frame_thres = fg.erode(con_frame_thres.astype(np.uint8), 7)  # amplify gaps/sepration between flies
                            con_frame_thres[con_frame_mask] = 0  # mask out adjecent flies/patches
                            con_centers, con_labels, con_points = fg.segment_cluster(con_frame_thres, num_clusters=flycnt[con])
                            # 4. if >2 flies in conn comp we watershed
                            if flycnt[con] > 2:  # only use additional watershed step when >2 flies in conn comp
                                con_frame[con_frame_mask] = 0  # mask out adjecent flies/patches
                                marker_positions = np.vstack(([0, 0], con_centers))  # "seeds" for the watershed - prepend [0,0] for background
                                con_centers, con_labels, con_points, _, _, ll = fg.segment_watershed(con_frame, marker_positions, frame_dilation=7)
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
                            points = points[labels[:, 0] != con, :]
                            labels = labels[labels[:, 0] != con]
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
                        centers[chb, :, :] = [np.median(points[labels[:, 0] == label, :], axis=0) for label in np.unique(labels)]
                    else:  # if still flies w/o conn compp fall back to segment_cluster
                        print(f"{flycnt[0]} outside of the conn comps or conn comp {np.where(flycnt[1:] == 0)} is empty - falling back to segment cluster - should mark frame as potential jump")
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
            # res.frame_error[res.frame_count, :, :] = frame_error
            res.area[res.frame_count, :] = 0
            yield res, foreground


def run(file_name, override=False, init_only=False, display=None, save_video=False, nflies=1, threshold=0.4, save_interval=1000, start_frame=None, led_coords=[10, 550, 100, -1]):
    """Track movie."""
    try:
        printf = lambda string: print(os.path.basename(file_name) + ": " + string)
        printf('processing ' + file_name)
        vr = VideoReader(file_name)

        if not override:
            try:  # attempt resume from intermediate results
                res = Results(file_name=os.path.normpath(file_name[:-4].replace('\\', '/') + '.h5'))
                res_loaded = True
                if start_frame is None:
                    start_frame = res.frame_count
                else:
                    res.frame_count = start_frame
                printf('resuming from {0}'.format(start_frame))
            except KeyboardInterrupt:
                raise
            except Exception as e:  # if fails start from scratch
                res_loaded = False
                pass

        if override or not res_loaded:  # re-initialize tracker
            if start_frame is None:
                start_frame = 0
            printf("start initializing")
            res = init(vr, start_frame, threshold, nflies, file_name, )
            printf("done initializing")
            pass

        if len(led_coords) != 4:
            ret, frame = vr.read()
            led_coords = fg.detect_led(frame)
            vr.reset()

        if init_only:
            return
        res.threshold = threshold
        vr.seek(start_frame)
        if save_video:
            ret, frame = vr.read()
            frame_size = tuple(np.uint(16 * np.floor(np.array(frame.shape[0:2], dtype=np.double) / 16)))
            vr.reset()
            printf('since x264 frame size need to be multiple of 16, frames will be truncated from {0} to {1}'.format(frame.shape[0:2], frame_size))
            vw = cv2.VideoWriter(file_name[0:-4] + "tracks.avi", fourcc=cv2.VideoWriter_fourcc(*'X264'),
                                 fps=vr.frame_rate, frameSize=frame_size)
        # iterate over frames
        start = time.time()
        ret = True
        frame_processor = Prc(res)

        while(ret and res.frame_count < res.number_of_frames + 1001):
            ret, frame = vr.read()
            if not ret:
                printf("frame returned False")
            else:
                res, foreground = frame_processor.prc(frame, res)

                res.led[res.frame_count] = np.mean(fg.crop(frame[:, :, 0], led_coords))
                # get annotated frame if necessary
                if save_video or (display is not None and res.frame_count % display == 0):
                    chamberID = 0  # fix to work with multiple chambers
                    # frame_with_tracks = cv2.cvtColor(np.uint8(fg.crop(foreground, np.ravel(res.chambers_bounding_box[chamberID+1][:, ::-1]))), cv2.COLOR_GRAY2RGB).astype(np.float32)
                    frame_with_tracks = cv2.cvtColor(np.uint8(fg.crop(frame[:, :, 0], np.ravel(res.chambers_bounding_box[chamberID+1][:, ::-1]))), cv2.COLOR_GRAY2RGB).astype(np.float32)/255.0
                    frame_with_tracks = fg.annotate(frame_with_tracks,
                                                    centers=np.clip(np.uint(res.centers[res.frame_count, chamberID, :, :]), 0, 10000),
                                                    lines=np.clip(np.uint(res.lines[res.frame_count, chamberID, 0:res.lines.shape[2], :, :]), 0, 10000))
                # display annotated frame
                if display is not None and res.frame_count % display == 0:
                    cv2.destroyAllWindows()
                    fg.show(frame_with_tracks, window_name=f"{res.frame_count}")

                # save annotated frame to video
                if save_video:
                    vw.write(np.uint8(frame_with_tracks[:frame_size[0], :frame_size[1], :]))

                if res.frame_count % 1000 == 0:
                    printf('frame {0} processed in {1:1.2f}.'.format(res.frame_count, time.time() - start))
                    start = time.time()

                if res.frame_count % save_interval == 0:
                    res.status = "progress"
                    res.save(file_name[0:-4] + '.h5')
                    printf("    saving intermediate results")

        # save results and clean up
        printf("finished processing frames - saving results")
        res.status = "done"
        res.save(file_name[0:-4] + '.h5')
        printf("             done.")
        return 1
    except KeyboardInterrupt:
        raise
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        printf(repr(traceback.extract_tb(exc_traceback)))
        ee = e
        print(ee)
        return 0
    finally:  # clean up - will be called before return statement
        if display is not None:  # close any windows
            cv2.destroyAllWindows()
        if save_video:
            vw.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='video file to process')
    parser.add_argument('--nflies', type=int, default=1, help='number of flies in video')
    parser.add_argument('-d', '--display', type=int, default=None, help='show every Nth frame')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='threshold for foreground detection, defaults to 0.3')
    parser.add_argument('-s', '--start_frame', type=float, default=None, help='first frame to track, defaults to 0')
    parser.add_argument('-o', '--override', action='store_true', help='override existing initialization or intermediate results')
    parser.add_argument('--init_only', action='store_true', help='only initialize, do not track')
    parser.add_argument('--save_video', action='store_true', help='save annotated vid with tracks')
    parser.add_argument('--led_coords', nargs='+', type=int, default=[10, 550, 100, -1], help='should be a sequence of 4 values OTHERWISE will autodetect')
    args = parser.parse_args()

    print('Tracking {0} flies in {1}.'.format(args.nflies, args.file_name))
    run(args.file_name, init_only=args.init_only, override=args.override, display=args.display, save_video=args.save_video,
        nflies=args.nflies, threshold=args.threshold, start_frame=args.start_frame, led_coords=args.led_coords)
