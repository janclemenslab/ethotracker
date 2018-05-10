import pytest
from tracker.background import BackGround, BackGroundMean, BackGroundMax, BackGroundMedian
from videoreader import VideoReader


root = '/Volumes/ukme04/#Common/playback/dat/'
filename = root + '/rpi8-20180206_185652/rpi8-20180206_185652.mp4'
num_bg_frames = 5


def test_max():
    with VideoReader(filename) as vr:
        bg = BackGroundMax(vr).estimate(num_bg_frames=num_bg_frames)


def test_median():
    with VideoReader(filename) as vr:
        bg = BackGroundMedian(vr).estimate(num_bg_frames=num_bg_frames)


def test_mean():
    with VideoReader(filename) as vr:
        bg = BackGroundMean(vr).estimate(num_bg_frames=num_bg_frames)


def test_exc():
    with pytest.raises(NotImplementedError):
        with VideoReader(filename) as vr:
            bg = BackGround(vr)
            bg.estimate(num_bg_frames=num_bg_frames)
