import numpy as np
import cv2
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

import matplotlib.pyplot as plt
plt.ion()

def init(vr, start_frame, threshold, nflies, file_name, num_bg_frames=1000):
    res = Results()                     # init results object
    bg = BackGround(vr)
    bg.estimate(num_bg_frames, start_frame)
    res.background = bg.background[:, :, 0]
    vr.reset()
    # detect chambers
    res.chambers = np.ones_like(res.background).astype(np.uint8)#fg.get_chambers(res.background, chamber_threshold=1.0, min_size=20000, max_size=200000, kernel_size=7)
    # res.chambers[:10,:] = 0


    res.chambers[:40,:] = 0
    # printf('found {0} chambers'.format( np.unique(res.chambers).shape[0]-1 ))

    # detect empty chambers
    # vr.seek(start_frame)
    # # 1. read frame and get foreground
    # ret, frame = vr.read()
    # foreground = fg.threshold(res.background - frame[:, :, 0], threshold * 255)
    # # fg.show(foreground)
    # # 2. segment and get flies and remove chamber if empty or "fly" too small
    # labels = np.unique(res.chambers)
    # area = np.array([fg.segment_center_of_mass(foreground * (res.chambers == label))[4] for label in labels])  # get fly size for each chamber
    # labels[area < 20] = 0                                                  # mark empty chambers for deletion
    # res.chambers, _, _ = fg.clean_labels(res.chambers, labels, force_cont=True)  # delete empty chambers
    res.chambers_bounding_box = fg.get_bounding_box(res.chambers)  # get bounding boxes of remaining chambers
    res.chambers_bounding_box[0] = [[0,0],[res.background.shape[0], res.background.shape[1]]]

    # init Results structure
    res.nflies = int(nflies)
    res.nchambers = int(np.max(res.chambers))
    res.file_name = file_name
    res.start_frame = int(start_frame)
    res.frame_count = int(start_frame)
    res.number_of_frames = int(vr.number_of_frames)
    if res.number_of_frames<=0:
        print('neg. number of frames detected - fallback to 3h recording')
        res.number_of_frames = vr.frame_rate * 60 * 60 * 3  # fps*sec*min*hour

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
            foreground = fg.erode(foreground.astype(np.uint8), kernel_size=4)
            foreground = cv2.medianBlur(foreground.astype(np.uint8), 3)  # get rid of specks
            for ii in uni_chambers:
                if ii > 0:  # 0 is background
                    foreground_cropped = foreground[chamber_slices[ii]] * (res.chambers[chamber_slices[ii]] == ii)  # crop frame to current chamber
                    if old_centers is None:  # on first pass get initial values - this only works if first frame produces the correct segmentation
                        centers[ii-1, :, :], labels, points,  = fg.segment_cluster(foreground_cropped, num_clusters=res.nflies)
                    else:  # for subsequent frames use connected components and split those with multiple flies
                        # if any flies outside of any component - grow blobs and repeat conn comps
                        cnt = 0  # n-repeat of conn comp
                        flycnt = [res.nflies] # init with all flies outside of conn comps
                        max_repeats = 1  # only do this once
                        while flycnt[0]>0 and cnt<max_repeats:
                            if cnt is not 0:  # do not display on first pass
                                print(f"{flycnt[0]} outside of the conn comps - growing blobs")
                                if flycnt[0]==res.nflies:
                                    print('   something is wrong')
                            foreground_cropped = fg.dilate(foreground_cropped.astype(np.uint8), kernel_size=5)
                            this_centers, this_labels, points, _, this_size, labeled_frame = fg.segment_connected_components(
                                                                                                foreground_cropped, minimal_size=5)
                            # get conn comp each fly is in using previous position
                            fly_conncomps = labeled_frame[np.uintp(old_centers[ii-1,:,0]), np.uintp(old_centers[ii-1,:,1])]
                            # count number of flies per conn comp
                            flycnt, flybins = np.histogram(fly_conncomps, bins=-0.5+ np.arange(np.max(labeled_frame+2)))
                            cnt += 1
                        if flycnt[0]==0:  # if all flies are assigned a conn comp - proceed
                            # bins -> conn comp ids
                            flybins = np.uintp(flybins[1:]-0.5)
                            # print(fly_conncomps)
                            # print(flybins)
                            # print(flycnt)

                            this_labels = np.reshape(this_labels, (this_labels.shape[0],1)) # make (n,1), not (n,) for compatibility downstream
                            labels = this_labels.copy()  # copy for new labels
                            # split conn compts with multiple flies using clustering
                            for con in np.uintp(flybins[flycnt>1]):
                                # cluster points for current conn comp
                                con_frame = labeled_frame==con
                                # erode to increase separation between flies in a blob
                                # con_frame = fg.erode(con_frame.astype(np.uint8), kernel_size=5)
                                con_centers, con_labels, con_points = fg.segment_cluster(con_frame, num_clusters=flycnt[con])
                                # with erosion:
                                # # delete old labels and points - if we erode we will have fewer points
                                # points = points[labels[:,0]!=con,:]
                                # labels = labels[labels[:,0]!=con]
                                # # append new labels and points
                                # labels = np.append(labels, np.max(labels)+10+con_labels, axis=0)
                                # points = np.append(points, con_points, axis=0)
                                # w/o erosion:
                                labels[this_labels==con] = np.max(labels)+10+con_labels[:, 0]

                            # make labels consecutive numbers again
                            new_labels = np.zeros_like(labels)
                            for cnt, label in enumerate(np.unique(labels)):
                                new_labels[labels==label] = cnt
                            labels = new_labels.copy()
                            # if np.unique(labels).shape[0]>res.nflies:
                                # import ipdb; ipdb.set_trace()
                                # plt.imshow(labeled_frame);plt.plot(old_centers[ii-1,:,1], old_centers[ii-1,:,0], '.r')
                                # plt.scatter(points[:,1], points[:,0], c=labels[:,0])
                            # calculate center values from new labels
                            for label in np.unique(labels):
                                centers[ii-1, label, :] = np.median(points[labels[:,0]==label,:], axis=0)
                        else:  # if still flies w/o conn compp fall back to segment_cluster
                            print(f"{flycnt[0]} outside of the conn comps - falling back to segment cluster - should mark frame as potential jump")
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


def run(file_name, override=False, init_only=False, display=None, save_video=False, nflies=1, threshold=0.4, save_interval=1000, start_frame=None, led_coords=[10, 550, 100, -1]):
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

        if len(led_coords)!=4:
            ret, frame = vr.read()
            led_coords = fg.detect_led(frame)
            vr.reset()
        print(led_coords)
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
                        # frame_with_tracks = fg.annotate(cv2.cvtColor(np.uint8(foreground), cv2.COLOR_GRAY2RGB),
                        #                                 centers=np.clip(np.uint(res.centers[res.frame_count, 0, :, :]),0,10000),
                        #                                 lines=np.clip(np.uint(res.lines[res.frame_count, 0, 0:res.lines.shape[1], :, :]),0,10000))
                        chamberID = 0 # fix to work with multiple chambers

                        frame_with_tracks = fg.annotate(frame[80:,:,:]/255,
                                                    centers=np.clip(np.uint(res.centers[res.frame_count, chamberID, :, :]),0,10000),
                                                    lines=np.clip(np.uint(res.lines[res.frame_count, chamberID, 0:res.lines.shape[2], :, :]),0,10000))
                        # frame_with_tracks = fg.annotate(cv2.cvtColor(np.uint8(foreground[80:,:]), cv2.COLOR_GRAY2RGB).astype(np.float32),
                        #                                 centers=np.clip(np.uint(res.centers[res.frame_count, chamberID, :, :]),0,10000),
                        #                                 lines=np.clip(np.uint(res.lines[res.frame_count, chamberID, 0:res.lines.shape[2], :, :]),0,10000))

                    # display annotated frame
                    if display is not None and res.frame_count % display == 0:
                        # fg.show(np.float32(frame_with_tracks)/255.0, autoscale=False)
                        cv2.destroyAllWindows()
                        fg.show(frame_with_tracks, window_name=f"{res.frame_count}")

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
    parser.add_argument('-s', '--start_frame', type=float, default=0, help='first frame to track, defaults to 0')
    parser.add_argument('-o', '--override', action='store_true', help='override existing initialization or intermediate results')
    parser.add_argument('--init_only', action='store_true', help='only initialize, do not track')
    parser.add_argument('--save_video', action='store_true', help='save annotated vid with tracks')
    parser.add_argument('--led_coords', nargs='+', type=int, default=[10, 550, 100, -1], help='should be a sequence of 4 values OTHERWISE will autodetect')
    args = parser.parse_args()

    print('Tracking {0} flies in {1}.'.format(args.nflies, args.file_name))
    run(args.file_name, init_only=args.init_only, override=args.override, display=args.display, save_video=args.save_video,
        nflies=args.nflies, threshold=args.threshold, start_frame=args.start_frame, led_coords=args.led_coords)
