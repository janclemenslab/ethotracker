"""Track videos."""
import cv2
import argparse
import time
import sys
import traceback
import os
import logging
import numpy as np

from tracker.videoreader import VideoReader
from tracker.attrdict import AttrDict
import tracker.foreground as fg


def annotate_frame(frame, res, raw_frame=True):
    """Add centroids and lines to frame."""
    uni_chambers = np.unique(res.chambers).astype(np.int)
    frame_with_tracks = list()
    for chamberID in range(np.max(uni_chambers)):
        if raw_frame:
            tmp = fg.crop(frame[:, :, 0], np.ravel(res.chambers_bounding_box[chamberID+1][:, ::-1]))
            tmp = cv2.cvtColor(np.uint8(tmp), cv2.COLOR_GRAY2RGB)
            tmp = tmp.astype(np.float32)/255.0
        else:  # otherwise use background subtracted and thresholded frames
            tmp = fg.crop(frame, np.ravel(res.chambers_bounding_box[chamberID+1][:, ::-1]))
            tmp = cv2.cvtColor(np.uint8(tmp), cv2.COLOR_GRAY2RGB)
            tmp = tmp.astype(np.float32)
        tmp = fg.annotate(tmp,
                          centers=np.clip(np.uint(res.centers[res.frame_count, chamberID, :, :]), 0, 10000),
                          lines=np.clip(np.uint(res.lines[res.frame_count, chamberID, 0:res.lines.shape[2], :, :]), 0, 10000))
        frame_with_tracks.append(tmp)
    return frame_with_tracks


def display_frame(frame_with_tracks):
    """Display list of frames."""
    import matplotlib.pyplot as plt
    plt.ion()
    plt.clf()
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    for cnt, frame in enumerate(frame_with_tracks):
        plt.subplot(1, len(frame_with_tracks), cnt+1)
        plt.imshow(frame)
        plt.axis('off')
    # plt.tight_layout()
    plt.show()
    plt.pause(0.0001)


def run(file_name, override=False, init_only=False, display=None, save_video=False, nflies=1,
        threshold=0.4, save_interval=1000, start_frame=None, led_coords=[10, 550, 100, -1],
        processor='chaining'):
    """Track movie."""
    if processor == 'chaining':
        from tracker.frame_processor_chaining import Prc, init
    elif processor == 'playback':
        from tracker.frame_processor_playback import Prc, init
    else:
        raise TypeError(f'Unknown frame processor type {processor}. Should be `chaining` or `playback`.')

    logging.info(f'processing {file_name}')
    vr = VideoReader(file_name)

    if not override:
        try:  # attempt resume from intermediate results
            res = AttrDict().load(filename=os.path.normpath(file_name[:-4].replace('\\', '/') + '.h5'))
            res_loaded = True
            if start_frame is None:
                start_frame = res.frame_count
            else:
                res.frame_count = start_frame
            logging.info(f'resuming from {start_frame}')
        except KeyboardInterrupt:
            raise
        except Exception as e:  # if fails start from scratch
            print(e)
            res_loaded = False
            pass

    if override or not res_loaded:  # re-initialize tracker
        if start_frame is None:
            start_frame = 0
        logging.info('start initializing')
        res = init(vr, start_frame, threshold, nflies, file_name, )
        logging.info('done initializing')
        vr = VideoReader(file_name)  # for some reason need to re-intantiate here - otherwise returns None frames


    if len(led_coords) != 4:
        led_coords = fg.detect_led(vr[res.start_frame])

    if init_only:
        return

    res.threshold = threshold
    if save_video:
        frame_size = tuple(np.uint(16 * np.floor(np.array(vr[0].shape[0:2], dtype=np.double) / 16)))
        logging.warn('since x264 frame size need to be multiple of 16, frames will be truncated from {0} to {1}'.format(vr[0].shape[0:2], frame_size))
        vw = cv2.VideoWriter(file_name[0:-4] + "tracks.avi", fourcc=cv2.VideoWriter_fourcc(*'X264'),
                             fps=vr.frame_rate, frameSize=frame_size)

    frame_processor = Prc(res)

    # iterate over frames
    start = time.time()
    for frame in vr[start_frame:]:
        try:
            res, foreground = frame_processor.process(frame, res)
            res.led[res.frame_count] = np.mean(fg.crop(frame, led_coords))

            # get annotated frame if necessary
            if save_video or (display is not None and res.frame_count % display == 0):
                frame_with_tracks = annotate_frame(frame, res, raw_frame=True)

            # display annotated frame
            if display is not None and res.frame_count % display == 0:
                display_frame(frame_with_tracks)

            # save annotated frame to video
            if save_video:
                vw.write(np.uint8(frame_with_tracks[:frame_size[0], :frame_size[1], :]))

            if res.frame_count % save_interval == 0:
                logging.info('frame {0} processed in {1:1.2f}.'.format(res.frame_count, time.time() - start))
                start = time.time()

            if res.frame_count % save_interval == 0:
                res.status = "progress"
                res.save(file_name[0:-4] + '.h5')
                logging.info("    saving intermediate results")
        except KeyboardInterrupt:
            raise
        except Exception as e:  # catch errors during frame processing
            logging.error(f'---- error processing frame {res.frame_count} ----')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            logging.error(repr(traceback.extract_tb(exc_traceback)))
            ee = e
            logging.error(ee)
            # clean up - will be called before
            if save_video:
                vw.release()
            del(vr)
            return -1

    # save results and clean up
    logging.info("finished processing frames - saving results")
    res.status = "done"
    res.save(file_name[0:-4] + '.h5')
    logging.info("             done.")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='video file to process')
    parser.add_argument('--nflies', type=int, default=1, help='number of flies in video')
    parser.add_argument('-d', '--display', type=int, default=None, help='show every Nth frame')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='threshold for foreground detection, defaults to 0.3')
    parser.add_argument('-s', '--start_frame', type=int, default=None, help='first frame to track, defaults to 0')
    parser.add_argument('-o', '--override', action='store_true', help='override existing initialization or intermediate results')
    parser.add_argument('-p', '--processor', type=str, choices=['chaining', 'playback'], default='chaining', help='class to process frames')
    parser.add_argument('--init_only', action='store_true', help='only initialize, do not track')
    parser.add_argument('--save_video', action='store_true', help='save annotated vid with tracks')
    parser.add_argument('--led_coords', nargs='+', type=int, default=[10, 550, 100, -1], help='should be a sequence of 4 values OTHERWISE will autodetect')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info('Tracking {0} flies in {1}.'.format(args.nflies, args.file_name))

    run(args.file_name, init_only=args.init_only, override=args.override, display=args.display, save_video=args.save_video,
        nflies=args.nflies, threshold=args.threshold, start_frame=args.start_frame, led_coords=args.led_coords, processor=args.processor)
