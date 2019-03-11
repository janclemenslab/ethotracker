"""A family of classes for estimating backgrounds."""
import numpy as np
import cv2
import argparse
import os
cv2.setNumThreads(0)

class BackGround():
    """Calculate background as mean over frames - abstract base class.

    ARGS
        vr - VideoReader instance
    VARS
        background       - background estimate (avg frame)
        background_count - number of frames which have been averaged to get `background`
    METHODS
        estimate(num_bg_frames=100) - estimate background from `num_bg_frames` covering whole video
        save(file_name) - save `background` to file_name.PNG
        load(file_name) - load `background` from file_name (uses cv2.imread)
    """

    def __init__(self, vr):
        """Construct - vr is VideoReader instance."""
        self.vr = vr
        self.frames = []
        self.background = np.zeros((self.vr.frame_width, self.vr.frame_height, self.vr.frame_channels))
        self.background_count = 0

    def estimate(self, num_bg_frames=100, start_frame=1):
        """Estimate back ground from video.

        `_accumulate` and then `_normalize` frames - these can be overridden.
        Args:
            num_bg_frames: number of (evenly spaced) frames (spannig whole video) over which to average (defaut 100)
            start_frame: first frame to read
        Returns:
            background
        """
        frame_interval = int(len(self.vr)/num_bg_frames)
        stop_frame = len(self.vr)
        for frame in self.vr[int(start_frame):stop_frame:frame_interval]:
            if frame is not None:
                self._accumulate(frame)
        self._normalize()
        return self.background

    def _accumulate(self, frame):
        """Update background with `frame`."""
        raise NotImplementedError

    def _normalize(self):
        raise NotImplementedError

    def save(self, file_name):
        """Save `background` as file_name.PNG."""
        return cv2.imwrite(file_name, self.background)

    def save_binary(self, file_name):
        """Save `background` as file_name.NPY."""
        return np.savez(file_name, self.background)

    def load(self, file_name):
        """Load `background` from file_name."""
        self.background = cv2.imread(file_name)


class BackGroundMean(BackGround):
    """Calculate background as mean over frames."""

    def _accumulate(self, frame):
        """Update background with `frame`."""
        cv2.accumulate(frame, self.background)
        self.background_count += 1

    def _normalize(self):
        self.background = self.background / self.background_count


class BackGroundMax(BackGround):
    """Calculate background as maximum over frames."""

    def _accumulate(self, frame):
        """Update background (mean frame) with `frame`."""
        self.background = np.maximum(frame, self.background)  # element-wise max operation
        self.background_count += 1

    def _normalize(self):
        # nothing to do - overwrite parent function with this empty one
        pass


class BackGroundMedian(BackGround):
    """Calculate background as median over frames. Memory intensive."""

    def _accumulate(self, frame):
        """_accumulates background (mean frame) with `frame`."""
        self.frames.append(frame)
        self.background_count += 1

    def _normalize(self):
        self.background = np.nanmedian(self.frames, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='video file to process')
    parser.add_argument('-n', '--num_bg_frames', type=int, default=100, help='number of frames for estimating background (100)')
    parser.add_argument('-f', '--format', type=str, default='png', help='image format for background (png)')
    parser.add_argument('-s', '--savebin', action='store_true', help='save as binary matrix (npy-format)')
    parser.add_argument('-t', '--type', action='store', choices=['mean', 'max', 'median'], default='mean')

    args = parser.parse_args()
    print(args)

    from tracker.VideoReader import VideoReader
    with VideoReader(args.filename) as vr:
        if args.type == 'max':
            bg = BackGroundMax(vr)
        elif args.type == 'median':
            bg = BackGroundMedian(vr)
        else:
            bg = BackGround(vr)

        bg.estimate(num_bg_frames=args.num_bg_frames)

        if args.format is not None:
            print('saving background image as {base}.{format}'.format(base=os.path.splitext(args.filename)[0], format=args.format))
            bg.save('{base}.{format}'.format(base=os.path.splitext(args.filename)[0], format=args.format))

        if args.savebin:
            print('saving raw background to {base}.npy'.format(base=os.path.splitext(args.filename)[0]))
            bg.save_binary('{base}.npy'.format(base=os.path.splitext(args.filename)[0]))
