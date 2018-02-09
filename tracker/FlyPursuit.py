from __future__ import print_function
import numpy as np

import argparse
# import pyprind
from multiprocessing import Process, current_process
import time
import sys
import traceback
import os

from .VideoReader import VideoReader
from .BackGround import BackGround
from .Results import Results
import tracker.ForeGround as fg
import tracker.Tracker as tk


def init(vr, start_frame, threshold, nflies, file_name, num_bg_frames=1000):
    res = Results()                     # init results object

    # bg_loaded = False
    # if instance(num_bg_frames, str):
    #     try:
    #         background = cv2.imread(num_bg_frames)
    #         background = bg.background[:, :, 0]
    #         bg_loaded = True
    #     except Exception as e:
    #         print('could not load {file}'.format(num_bg_frames))
    #         print(e)
    #
    # if not bg_loaded:
    #     # printf('estimating background')

    bg = BackGround(vr)
    bg.estimate(num_bg_frames)
    res.background = bg.background[:, :, 0]
    vr.reset()

    # detect chambers
    res.chambers = fg.get_chambers(res.background, chamber_threshold=1.0, min_size=20000, max_size=200000, kernel_size=7)
    # printf('found {0} chambers'.format( np.unique(res.chambers).shape[0]-1 ))

    # detect empty chambers
    vr.seek(start_frame)
    # 1. read frame and get foreground
    ret, frame = vr.read()
    foreground = fg.threshold(res.background - frame[:, :, 0], threshold * 255)
    # fg.show(foreground)
    # 2. segment and get flies and remove chamber if empty or "fly" too small
    labels = np.unique(res.chambers)
    area = np.array([fg.segment_center_of_mass(foreground * (res.chambers == label))[4] for label in labels])  # get fly size for each chamber
    labels[area < 20] = 0                                                  # mark empty chambers for deletion
    res.chambers, _, _ = fg.clean_labels(res.chambers, labels, force_cont=True)  # delete empty chambers
    res.chambers_bounding_box = fg.get_bounding_box(res.chambers)  # get bounding boxes of remaining chambers


    # init Results structure
    res.nflies = int(nflies)
    res.nchambers = int(np.max(res.chambers))
    res.file_name = file_name
    res.start_frame = int(start_frame)
    res.frame_count = int(start_frame)
    res.number_of_frames = int(vr.number_of_frames)
    res.centers = np.zeros((res.number_of_frames + 1000, res.nchambers, 2), dtype=np.float16)
    res.area = np.zeros((res.number_of_frames + 1000, res.nchambers), dtype=np.float16)
    res.lines = np.zeros((res.number_of_frames + 1000, res.nchambers, 2, 2), dtype=np.float16)
    res.led = np.zeros((res.number_of_frames + 1000, 1), dtype=np.float16)
    # save initialized results object
    res.status = "initialized"
    res.save(file_name=file_name[0:-4] + '.h5')
    # printf('saving init')
    print('found {0} fly bearing chambers'.format( res.nchambers ))
    return res


class Prc():
    def __init__(self, res):
        self.frame_processor = self.process(res)

    def prc(self, frame, res):
        next(self.frame_processor)
        res, foreground = self.frame_processor.send((frame, res))
        return res, foreground

    def process(self, res):
        # init
        old_centers = None
        centers = np.zeros((res.nchambers, 2))
        old_lines = None
        lines = np.zeros((res.nchambers, 2, 2))
        area = np.zeros((1, res.nchambers))
        uni_chambers = np.unique(res.chambers).astype(np.int)
        chamber_slices = [None] * int(res.nchambers + 1)
        for ii in uni_chambers:
            chamber_slices[ii] = (np.s_[res.chambers_bounding_box[ii, 0, 0]:res.chambers_bounding_box[ii, 1, 0],
                                  res.chambers_bounding_box[ii, 0, 1]:res.chambers_bounding_box[ii, 1, 1]])
        # process
        while True:
            frame, res = yield
            res.frame_count = int(res.frame_count + 1)
            foreground = fg.threshold(res.background - frame[:, :, 0], res.threshold * 255)
            for ii in uni_chambers:
                if ii > 0:  # 0 is background
                    foreground_cropped = foreground[chamber_slices[ii]] * (res.chambers[chamber_slices[ii]] == ii)  # crop frame to current chamber
                    centers[ii - 1, :], _, points, _, area[0, ii - 1] = fg.segment_center_of_mass(foreground_cropped)
                    if points.shape[0] > 0:   # in case we lose a fly
                        lines[ii - 1, :, :], _ = tk.fit_line(points)

            if res.nflies > 1 and old_centers is not None:  # match centers across frames - not needed for one fly per chamber
                new_labels, centers = tk.match(old_centers, centers)
            old_centers = centers  # remember

            if old_lines is not None:  # fix forward/backward flips
                lines, is_flipped, D = tk.fix_flips(old_lines, lines)
            old_lines = lines  # remember

            res.centers[res.frame_count, :, :] = centers
            res.lines[res.frame_count, 0:lines.shape[0], :, :] = lines
            res.area[res.frame_count, :] = area
            yield res, foreground


def run(file_name, override=False, init_only=False, display=None, save_video=False, nflies=1, threshold=0.4, save_interval=1000, start_frame=50, led_coords=[10, 550, 100, -1]):
    try:
        printf = lambda string: print(os.path.basename(file_name) + ": " + string)
        printf('processing ' + file_name)
        vr = VideoReader(file_name)

        if not override:
            try:  # attempt resume from intermediate results
                res = Results(file_name=os.path.normpath(file_name[:-4].replace('\\', '/') + '.h5'))
                res_loaded = True
                printf('resuming from {0}'.format(res.frame_count))

            except Exception as e:  # if fails start from scratch
                res_loaded = False
                pass

        if override or not res_loaded:  # re-initialize tracker
            printf("start initializing")
            res = init(vr, start_frame, threshold, nflies, file_name, )
            printf("done initializing")
            pass

        if init_only:
            return
        res.threshold = threshold
        vr.seek(res.frame_count)
        if save_video:
            frame_size = tuple(np.uint(16 * np.floor(np.array(vr.frame.shape[0:2], dtype=np.double) / 16)))
            printf('since x264 frame size need to be multiple of 16, frames will be truncated from {0} to {1}'.format(vr.frame.shape[0:2], frame_size))
            vw = cv2.VideoWriter(file_name[0:-4] + "tracks.avi", fourcc=cv2.VideoWriter_fourcc(*'X264'),
                                 fps=vr.frame_rate, frameSize=frame_size)

        # iterate over frames
        start = time.time()
        ret = True
        frame_processor = Prc(res)

        while(ret and res.frame_count < vr.number_of_frames + 1001):
            ret, frame = vr.read()
            if not ret:
                printf("frame returned False")
            else:
                try:
                    res, foreground = frame_processor.prc(frame, res)

                    res.led[res.frame_count] = np.mean(fg.crop(frame[:, :, 0], led_coords))
                    # get annotated frame if necessary
                    if save_video or (display is not None and res.frame_count % display == 0):
                        frame_with_tracks = fg.annotate(cv2.cvtColor(np.uint8(foreground), cv2.COLOR_GRAY2RGB),
                                                        centers=np.clip(np.uint(res.centers[res.frame_count, :, :]),0,10000),
                                                        lines=np.uint(res.lines[res.frame_count, 0:res.lines.shape[1], :, :]))

                    # display annotated frame
                    if display is not None and res.frame_count % display == 0:
                        fg.show(frame_with_tracks)

                    # save annotated frame to video
                    if save_video:
                        vw.write(np.uint8(frame_with_tracks[0:frame_size[0], 0:frame_size[1], :]))

                    if res.frame_count % 1000 == 0:
                        printf('frame {0} processed in {1:1.2f}.'.format(res.frame_count, time.time() - start))
                        start = time.time()

                    if res.frame_count % save_interval == 0:
                        res.status = "progress"
                        res.save(file_name[0:-4] + '.h5')
                        printf("    saving intermediate results")
                except Exception as e:  # catch errors during frame processing
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                    printf(repr(traceback.extract_tb(exc_traceback)))
                    ee = e
                    print(ee)

        # save results and clean up
        printf("finished processing frames - saving results")
        res.status = "done"
        res.save(file_name[0:-4] + '.h5')
        printf("             done.")
        if display is not None:  # close any windows
            cv2.destroyAllWindows()
        if save_video:
            vw.release()
        return 1

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        printf(repr(traceback.extract_tb(exc_traceback)))
        ee = e
        print(ee)
        if display is not None:  # close any windows
            cv2.destroyAllWindows()
        if save_video:
            vw.release()
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='video file to process')
    parser.add_argument('--nflies', type=int, default=1, help='number of flies in video')
    parser.add_argument('-d', '--display', type=int, default=None, help='show every Nth frame')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='threshold for foreground detection, defaults to 0.3')
    parser.add_argument('-s', '--start_frame', type=float, default=0, help='first frame to track, defaults to 50')
    parser.add_argument('-o', '--override', action='store_true', help='override existing initialization or intermediate results')
    parser.add_argument('--init_only', action='store_true', help='only initialize, do not track')
    parser.add_argument('--save_video', action='store_true', help='save annotated vid with tracks')
    args = parser.parse_args()

    print('Tracking {0} flies in {1}.'.format(args.nflies, args.file_name))
    run(args.file_name, init_only=args.init_only, override=args.override, display=args.display, save_video=args.save_video, nflies=args.nflies, threshold=args.threshold, start_frame=args.start_frame)
