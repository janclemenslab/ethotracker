import numpy as np
import os
import cv2
from contextlib import contextmanager


class VideoReader:
    """wrapper around opencv's VideoCapture()
        USAGE
            vr = VideoReader("video.avi", crop=[10, 20, 100, 80])
            vr.read
        ARGS
         file_name
         crop       - crop coordinates [x1,y1,x2,y2]
      VARS
         frame_width, frame_height, frame_channels, frame_rate, number_of_frames, fourcc
      METHODS
         read, reset, seek
         close
    """

    def __init__(self, file_name, crop=None):
        if not os.path.exist(file_name):
            raise FileNotFoundError
        self._file_name = file_name
        self._vr = cv2.VideoCapture()
        self._vr.open(self._file_name)
        self._crop = crop
        # read frame to test videoreader and get number of channels
        ret, frame = self.read()
        (self.frame_width, self.frame_height, self.frame_channels) = np.uintp(frame.shape)
        self.frame_shape = np.uintp(frame.shape)
        self.number_of_frames = int(self._vr.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self._vr.get(cv2.CAP_PROP_FPS))
        self.fourcc = int(self._vr.get(cv2.CAP_PROP_FOURCC))

    def __del__(self):
        self.close()

    # standard usage
    def read(self, frame_number=None):
        """read next frame or frame specified by `frame_number`"""
        if frame_number is not None:  # seek
            self.seek(frame_number)
        ret, frame = self._vr.read()  # read
        if ret and self._crop is not None:
            frame = frame[self._crop[1]:self._crop[3], self._crop[0]:self._crop[2], :]
        return ret, frame

    def reset(self):
        """re-initialize object"""
        self.__init__(self._file_name, crop=self._crop)

    def seek(self, frame_number):
        """go to frame"""
        self._vr.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def close(self):
        self._vr.release()


# as a context for working with `with` statements
@contextmanager
def video_file(path, crop=None):
    vr = VideoReader(path, crop)
    try:
        yield vr
    finally:
        vr.close()
        vr = None


# as a frame generator
def video_generator(path, crop=None):
    with video_file("test/160125_1811_1.avi") as video:
        while True:
            yield video.read()


def test():
    # standard usage
    print("testing as standard class")
    vr1 = VideoReader("test/160125_1811_1.avi")
    _, frame = vr1.read(100)
    cv2.imwrite("test/frame100.png", frame)


    # `with` statement
    print("testing as context")
    with video_file("test/160125_1811_1.avi") as vr2:
        for _ in range(2):
            ret, frame = vr2.read()
            print(frame.shape)

    print(vr2._vr.isOpened())

    # as generator
    print("testing as generator")
    vid_gen = video_generator("test/160125_1811_1.avi")
    for _ in range(2):
        ret, frame = next(vid_gen)
        print(frame.shape)

if __name__ == "__main__":
    test()
