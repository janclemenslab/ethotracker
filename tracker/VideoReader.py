"""Pythonic wrapper around opencv's VideoCapture()."""
import numpy as np
import os
import cv2


class VideoReader:
    """Pythonic wrapper around opencv's VideoCapture().

    USAGE
        vr = VideoReader("video.avi")  # initialize
        # use as Sequence
        vr[frame_number]
        vr[0:1000:10000]
        print(f'Video has {len(vr)} frames.')
        # use as generator/iterator
        for frame in vr:
            print(frame.shape)
        # or to specify start frame
        for frame in vr.frames(start_frame):
            print(frame.shape)
        # as context
        with VideoReader("video.avi") as vr:
            print(vr[0].shape)
    ARGS
        filename
    PROPERTIES
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
        """Now we can get frame via self[index] and self[start:stop:step]."""
        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]
        return self.read(index)[1]

    def __str__(self):
        return f"{self._filename} with {len(self)} frames at {self.frame_rate} fps"

    def __iter__(self):
        return self.frames(start=0, stop=None, step=1)

    def frames(self, start=0, stop=None, step=1):
        """Yield all frames (or a `slice` thereof).

        ARGS
            start=0
            stop=number of frames
            step=1
        YIELDS
            frame
        """
        self._seek(start)
        if stop is None:
            stop = len(self)
        for frame_number in range(start, stop, step):
            yield self._vr.read()[1]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read(self, frame_number=None):
        """Read next frame or frame specified by `frame_number`."""
        if frame_number is not None:  # seek
            self._seek(frame_number)
        ret, frame = self._vr.read()  # read
        return ret, frame

    def close(self):
        """Release video file."""
        self._vr.release()

    def _reset(self):
        """Re-initialize object."""
        self.__init__(self._filename)

    def _seek(self, frame_number):
        """Go to frame."""
        self._vr.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


def _test(filename="/Users/janc/rpi8-20180206_185652/rpi8-20180206_185652.mp4"):
    # standard usage
    print("testing as standard class")
    vr1 = VideoReader(filename)
    print(vr1)
    _, frame = vr1.read(100)
    print(frame.shape)
    # as sequence
    vrlist = VideoReader(filename)
    frame = vrlist[100]
    for frame in vrlist[::10000]:
        print(frame.shape)
    # as context
    print("testing as context")
    with VideoReader(filename) as vr2:
        print(vr2[0].shape)
    # as generator
    print("testing as generator")
    vr3 = VideoReader(filename)
    for frame in vr3.frames(start=100, stop=200, step=10):
        print(frame.shape)


if __name__ == "__main__":
    _test()
