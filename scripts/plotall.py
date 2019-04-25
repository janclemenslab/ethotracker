#!/usr/bin/env python3
import os
import shutil
import glob
import h5py
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# list all mp4 files
root = '/Volumes/ukme04/#Common/chaining'#'Z:\#Common\chaining'#'/Volumes/ukme04/#Common/playback'#'/scratch/clemens10/playback'#
datadir = f'{root}/dat'
logdir = f'{root}/log'
resdir = f'{root}/res'

print(datadir)
videofiles = [file for file in glob.glob(f'{datadir}/**/*.avi')]
# videofiles = [file for file in glob.glob(f'{resdir}/*spd.h5')]
videofiles.sort()
print(videofiles)

plt.ion()
plt.gcf().set_size_inches(30, 5)
# build and execute command
for videofile in videofiles:
    try:
        basename = os.path.splitext(os.path.basename(videofile))[0]
        basepath = os.path.splitext(videofile)[0]
        # shutil.copyfile(f"{basepath}.h5", f"{resdir}/{basename}.h5")
        plt.clf()
        print(f"{basepath}.h5, {basename}")

        with h5py.File(f"{resdir}/{basename}.h5", 'r') as f:
            centers = f['centers'][:]
        plt.plot(centers[:, 0, :, 0], linewidth=0.75)

        # with h5py.File(f"{resdir}/{basename}_spd.h5", 'r') as f:
        #     lines = f['lines_fixed'][:]
        # plt.plot(lines[:, 0, :, 1, 1], linewidth=0.75)

        plt.savefig(f"/Users/jclemens/plots/{basename}.png")
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x = centers[1000:10000, 0, :, 1]
        # y = centers[1000:10000, 0, :, 0]
        # z = np.arange(x.shape[0])/100
        # ax.plot(x ,y , z)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e)
