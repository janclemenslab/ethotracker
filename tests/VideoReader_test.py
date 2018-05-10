import pytest
from videoreader import VideoReader

root = '/Volumes/ukme04/#Common/playback/dat/'
filename = root + '/rpi8-20180206_185652/rpi8-20180206_185652.mp4'


def test_exc():
    with pytest.raises(FileNotFoundError) as excinfo:
        vr = VideoReader('thisFileDoesNotExist')


def test_repr():
    print(VideoReader(filename))


def test_func():
    _, frame = VideoReader(filename).read(100)
    print(frame.shape)


def test_func():
    print(VideoReader(filename)[100].shape)


def test_context():
    with VideoReader(filename) as vr:
        print(vr[0].shape)


def test_generator():
    with VideoReader(filename) as vr:
        for frame in vr[::50000]:
            print(frame.shape)
