"""Track videos."""
import platform
import sys
import traceback
import logging
import time
from enum import Enum
from typing import List
import os
import cv2

import numpy as np
import defopt

from videoreader import VideoReader
from attrdict import AttrDict
from . import foreground as fg


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


def display_frame(nb_chambers):
    """Display list of frames."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=nb_chambers, squeeze=False)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    RUN = True
    while RUN:
        frame_with_tracks = yield
        if frame_with_tracks is not None:
            for cnt, frame in enumerate(frame_with_tracks):
                ax[cnt,0].imshow(frame)
                ax[cnt,0].axis('off')
            fig.show()
            plt.pause(0.00001)
        else:
            RUN = False
    plt.close()


class ProcessorType(Enum):
    chaining = 'chaining'
    playback = 'playback'
    playback_red = 'playback_red'
    chaining_hires = 'chaining_hires'
    chaining_coarse = 'chaining_coarse'


def run(file_name: str, save_name: str, *, nflies: int=1, display: int=0, threshold: float=0.4,
        start_frame: int=None, override: bool=False, processor='chaining',
        init_only: bool=False, write_video: bool=False, led_coords: List[int]=[], interval_save: int=1000,
        show_raw_frame: bool=True) -> int:
    """Multi-animal tracker.

    Args:
      file_name(str): video file to process
      save_name(str): file to save results to
      nflies(int): number of flies in video
      display(int): show every Nth frame (do not show anything if 0)
      threshold(float): threshold for foreground detection, defaults to 0.4
      start_frame(int): first frame to track, defaults to 0
      override(bool): override existing initialization or intermediate results
      processor(ProcessorType): class to process frames
      init_only(bool): only initialize, do not track
      write_video(bool): save annotated vid with tracks
      led_coords(list[int]): should be a sequence of 4 values OTHERWISE will autodetect'
      interval_save(int): save intermediate resultse very nth frame

    Returns:
      exitcode

    """
    """Track movie."""
    if processor.value == 'chaining':
        from .frame_processor_chaining import Prc, init
    elif processor.value == 'playback':
        from .frame_processor_playback import Prc, init
    elif processor.value == 'playback_red':
        from .frame_processor_playback_red import Prc, init
    elif processor.value == 'chaining_hires':
        from .frame_processor_chaining_hires import Prc, init
    elif processor.value == 'chaining_coarse':
        from .frame_processor_chaining_coarse import Prc, init
    else:
        raise TypeError(f'Unknown frame processor type {processor}. Should be `chaining` or `playback`.')

    logging.info(f'processing {file_name}')
    vr = VideoReader(file_name)

    # __________ THIS SHOULD BE FACTORED OUT  _________________
    # also, fix the whole start_frame vs res.frame_count issue
    if not override:
        try:  # attempt resume from intermediate results
            res = AttrDict().load(save_name)
            res_loaded = True
            if start_frame is None:
                start_frame = res.frame_count
            else:
                res.frame_count = start_frame
            logging.info(f'resuming from {start_frame}')
        except KeyboardInterrupt:
            raise
        except Exception as e:  # if fails start from scratch
            logging.warning(e)
            logging.info('   will start from scratch.')
            res_loaded = False
            pass

    if override or not res_loaded:  # re-initialize tracker
        if start_frame is None:
            start_frame = 0
        logging.info('start initializing')
        res = init(vr, start_frame, threshold, nflies, file_name, )
        logging.info('done initializing')
        vr = VideoReader(file_name)  # for some reason need to re-intantiate here - otherwise returns None frames

    logging.info('Tracking {0} flies in {1}.'.format(res.nflies, file_name))

    # this should happen in frame processor for playback - not needed for chaining since we annotate
    if not hasattr(res, 'led_coords') or res.led_coords is None:
        logging.info('no leed coords in res')
        if len(led_coords) == 4:
            logging.info('using coords provided in arg')
            res.led_coords = led_coords
        else:
            logging.info('and no leed coords provided as arg - auto detecting')
            res.led_coords = fg.detect_led(vr[res.start_frame])

    if init_only:
        return

    res.threshold = threshold
    if write_video:
        frame_size = tuple(np.uint(32 * np.floor(np.array(vr[0].shape[0:2], dtype=np.double) / 32)))
        logging.warn('since x264 frame size need to be multiple of 16, frames will be truncated from {0} to {1}'.format(vr[0].shape[0:2], frame_size))
        vw = cv2.VideoWriter(file_name[0:-4] + "tracks.avi", fourcc=cv2.VideoWriter_fourcc(*'X264'),
                             fps=vr.frame_rate, frameSize=frame_size)

    frame_processor = Prc(res)

    # iterate over frames
    start = time.time()
    logging.info(f"Processing {res.number_of_frames} frames, starting at {res.start_frame}.")
    if display:
        disp = display_frame(res.nchambers)
        disp.send(None)
    for frame in vr[start_frame:]:
        try:
            res, foreground = frame_processor.process(frame, res)
            res.led[res.frame_count] = np.mean(fg.crop(frame, res.led_coords))
            # get annotated frame if necessary
            if write_video or (display and res.frame_count % display == 0):
                if show_raw_frame:
                    frame_with_tracks = annotate_frame(frame, res, raw_frame=True)
                else:
                    frame_with_tracks = annotate_frame(foreground, res, raw_frame=False)

            # display annotated frame
            if display and res.frame_count % display == 0:
                disp.send(frame_with_tracks)

            # save annotated frame to video
            if write_video:
                vw.write(np.uint8(frame_with_tracks[:frame_size[0], :frame_size[1], :]))

            if res.frame_count % interval_save == 0:
                logging.info('frame {0} processed in {1:1.2f}.'.format(res.frame_count, time.time() - start))
                start = time.time()

            if res.frame_count % interval_save == 0:
                res.status = "progress"
                res.save(save_name)
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
            # clean up
            if write_video:
                vw.release()
            del(vr)
            return -1

    # save results and clean up
    logging.info("finished processing frames - saving results")
    res.status = "done"
    res.save(save_name)
    logging.info("             done.")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    defopt.run(run)
