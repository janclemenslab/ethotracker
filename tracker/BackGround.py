import numpy as np
import cv2
from .VideoReader import VideoReader, video_file
import argparse
import os

class BackGround():
    """class for estimating background
       ARGS
          vr - VideoReader instance
       VARS
          background       - background estimate (avg frame)
          background_count - number of frames which have been averaged to get `background`
       METHODS
          estimate(num_bg_frames=100) - estimate background from `num_bg_frames` covering whole video
          update(frame) - background with frmae
          save(file_name) - save `background` to file_name.PNG
          load(file_name) - load `bckground` from file_name (uses cv2.imread)
    """

    def __init__(self, vr):
        """constructor - vr is VideoReader instance"""
        self.vr = vr
        self.background = np.zeros((self.vr.frame_width, self.vr.frame_height, self.vr.frame_channels))
        self.background_count = 0

    def estimate(self, num_bg_frames=100):
        """estimate back ground from video
              num_bg_frames - number of (evenly spaced) frames (spannig whole video) over which to average (defaut 100)
        """
        frame_numbers = np.linspace(1, self.vr.number_of_frames, num_bg_frames).astype(int)  # evenly sample movie
        for fr in frame_numbers:
            ret, frame = self.vr.read(fr)
            if ret and frame is not None:
                self.update(frame)
        self.background = self.background / self.background_count

    def update(self, frame):
        """updates background (mean frame) with `frame`"""
        cv2.accumulate(frame, self.background)
        self.background_count = self.background_count + 1

    def save(self, file_name):
        """save `background` as file_name.PNG for later reference"""
        return cv2.imwrite(file_name, self.background)

    def load(self, file_name):
        """load `background` from file_name"""
        try:
            self.background = cv2.imread(file_name)
        except Exception:
            pass


def test():
    vr = VideoReader("test/160125_1811_1.avi")
    bg = BackGround(vr)
    bg.estimate(500)
    bg.save("test/background.png")
    bg2 = BackGround(vr)
    bg2.load("test/background.png")
    print(bg2.background.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='video file to process')
    parser.add_argument('-n', '--num_bg_frames', type=int, default=100, help='number of frames for estimating background (100)')
    parser.add_argument('-f', '--format', type=str, default='tif', help='image format for background (png)')
    args = parser.parse_args()

    with video_file(args.filename) as vr:
        bg = BackGround(vr)
        bg.estimate(num_bg_frames=args.num_bg_frames)
        bg.save('{base}.{format}'.format(base=os.path.splitext(args.filename)[0], format=args.format))
