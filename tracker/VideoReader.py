import numpy as np
import os
import cv2
from contextlib import contextmanager


class VideoReader:
    """wrapper around opencv's VideoCapture()
        USAGE
            vr = VideoReader("video.avi")  # initialize

            # use as Sequence
            vr[frame_number]
            vr[0:1000:10000]
            print(f'Video has {len(vr)} frames.')

            # use as generator
            for frame in vr.frames():
                print(frame.shape)
        ARGS
         filename
        VARS
         frame_width, frame_height, frame_channels, frame_rate, number_of_frames, fourcc
      METHODS
         ret, frame = vr.read(framenumber): read frame, for compatibility with opencv VideoCapture
         vr.close(): release file
    """
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError
        self._filename = filename
        self._vr = cv2.VideoCapture()
        self._vr.open(self._filename)
        # read frame to test videoreader and get number of channels
        ret, frame = self.read()
        (self.frame_width, self.frame_height, self.frame_channels) = np.uintp(frame.shape)
        self.frame_shape = np.uintp(frame.shape)
        self.number_of_frames = len(self)
        self.frame_rate = int(self._vr.get(cv2.CAP_PROP_FPS))
        self.fourcc = int(self._vr.get(cv2.CAP_PROP_FOURCC))

    def __del__(self):
        self.close()

    def __len__(self):
        """Length is number of frames."""
        return int(self._vr.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, index):
        """Now we can get frame via self[index] and self[start:stop:step]"""
        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]
        return self.read(index)[1]

    def __str__(self):
        return f"{self._filename} with {len(self)} at {self.frame_rate} fps"

    def frames(self, start=0, stop=None, step=1):
        """generator that yields all frames"""
        self._seek(start)
        if stop==None:
            stop = len(self)
        # while True:
        for frame_number in range(start, stop, step):
            yield self._vr.read()[1]

    # def __enter__(self):
    #     return self
    #
    # def __exit__(self, *args):
    #     self.close()

    def read(self, frame_number=None):
        """Read next frame or frame specified by `frame_number`"""
        if frame_number is not None:  # seek
            self._seek(frame_number)
        ret, frame = self._vr.read()  # read
        return ret, frame

    def close(self):
        self._vr.release()

    def _reset(self):
        """re-initialize object"""
        self.__init__(self._filename)

    def _seek(self, frame_number):
        """go to frame"""
        self._vr.set(cv2.CAP_PROP_POS_FRAMES, frame_number)



# as a context for working with `with` statements
@contextmanager
def video_file(path):
    vr = VideoReader(path)
    try:
        yield vr
    finally:
        vr.close()
        vr = None


# as a frame generator
def video_generator(path):
    with video_file(path) as video:
        while True:
            yield video.read()


def test():
    # standard usage
    print("testing as standard class")
    vr1 = VideoReader("test/160125_1811_1.avi")
    _, frame = vr1.read(100)
    cv2.imwrite("test/frame100.png", frame)

    # as sequence
    vrlist = VideoReader("test/160125_1811_1.avi")
    frame100 = vrlist[100]
    for frame in vrlist[::10000]:
        print(frame.shape)

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
    pass# test()
