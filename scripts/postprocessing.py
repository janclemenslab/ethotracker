#!/usr/bin/env python3
import numpy as np
import h5py
import scipy.signal
import sys
import peakutils
from scipy import signal
import os

def load_data(file_name):
    with h5py.File(file_name, 'r') as f:
        pos = f['centers'][:]
        led = f['led'][:]
        start_frame = f['start_frame'].value
    pos = pos[start_frame + 1:-1000, :, :]
    led = led[start_frame + 1:-1000, 0].T
    nflies = pos.shape[1]
    return pos, led, nflies


def parse_prot(filename):
    with open(filename) as f:
        content = f.readlines()

    prot = {'timestamp': [], 'stimFileName': [], 'silencePre': [], 'silencePost': [], 'delayPost': [], 'intensity': [], 'freq': [], 'MODE': []}
    for line in content:
        token = line.split(' ')
        if len(token) > 1:  # consider only rows that contain protocol logs
            ts, id, *data = token
            prot['timestamp'].append(ts)
            prot_fields = data[0].split(';')
            for field in prot_fields[:-1]:
                key, value = field.split(',')
                prot[key].append(value)
    return prot


def get_led_peaks(led, thres=0.8, min_interval=0):
    led_diff = np.diff(signal.savgol_filter(led, 11, 6)).T
    # import ipdb; ipdb.set_trace()
    led_onsets = peakutils.indexes(led_diff[:,0], thres=thres)
    led_offsets = peakutils.indexes(-led_diff[:,0], thres=thres)

    # filter out repeats
    # prepend large value to intervals so we keep the first on- and offsets
    led_onset_interval = np.insert(np.diff(led_onsets), 0, 10e10)
    led_onsets = led_onsets[led_onset_interval > min_interval]

    led_offset_interval = np.insert(np.diff(led_offsets), 0, 10e10)
    led_offsets = led_offsets[led_offset_interval > min_interval]
    return led_onsets, led_offsets

def plot_led_peaks(led, led_onsets, led_offset, savefilename=None):
    import matplotlib.pyplot as plt
    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)

    ax = plt.subplot(311)
    ax.plot(led, linewidth=0.75)
    ax.set_title('LED trace')

    ax = plt.subplot(312)
    ax.plot(np.diff(led).T, linewidth=0.75)
    ax.plot(led_onsets,   4*np.ones_like(led_onsets), 'xr')
    ax.plot(led_offsets, -4*np.ones_like(led_offsets), 'xg')

    ax = plt.subplot(313)
    ax.plot(np.diff(led_onsets), 'o-')
    ax.plot(np.diff(led_offsets), 'x-')

    # plt.axis('tight')
    if savefilename:
        plt.savefig(savefilename)

def get_speed(pos, medfiltlen=None):
    spd = np.sqrt(np.sum(np.gradient(pos, axis=0).astype(np.float32)**2, axis=2))
    if medfiltlen is not None:
        spd = scipy.signal.medfilt(spd, medfiltlen)
    return spd


def chunk_data(data, cutpoints, chunklen):
    data_chunks = np.zeros((chunklen, 10000))
    cnt = 0
    for cutpoint in cutpoints:
        this = data[cutpoint:cutpoint + chunklen, :]
        data_chunks[:, cnt:cnt + this.shape[1]] = this
        cnt += this.shape[1]
    data_chunks = data_chunks[:, :cnt]  # keep only non-empty traces
    return data_chunks


def stats_by_group(data, grouping, fun):
    # TODO: pass through args and kwargs for fun

    # trim inputs
    data = data[:, :min(data.shape[1], grouping.shape[0])]
    grouping = grouping[:min(data.shape[1], grouping.shape[0])]

    group_labels, groups = np.unique(grouping, return_inverse=True)
    # pre-allocate
    group_stats = np.zeros((data.shape[0], len(group_labels)))
    # calc stat FUN for each group
    for idx, group in enumerate(np.unique(groups)):
        group_stats[:, idx] = fun(data[:, groups == group], axis=1)

    return group_labels, group_stats


if __name__ == '__main__':
    track_file_name = sys.argv[1]
    prot_file_name = sys.argv[2]
    save_file_name = sys.argv[3]
    print('processing tracks in {0} with playlist {1}. will save to {2}'.format(track_file_name, prot_file_name, save_file_name))

    # read tracking data
    pos, led, nflies = load_data(track_file_name)

    # detect LED onsets
    led_onsets, led_offsets = get_led_peaks(led, thres=0.8, min_interval=1000)
    try:
        plot_led_peaks(led, led_onsets, led_offsets, os.path.splitext(save_file_name)[0]+'.png')
    except:
        pass
    if len(led_onsets):
        print('found {0} led onsets'.format(len(led_onsets)))
        spd = get_speed(pos[:,:,0,:], 7)
        # chunk data
        chunklen = 4000
        chunkpre = 2000
        trial_traces = chunk_data(spd, led_onsets[:-1] - chunkpre, chunklen)
        # calc base line and test spd
        spd_test = np.nanmean(trial_traces[2000:2400, :], axis=0)
        spd_base = np.nanmean(trial_traces[1000:1800, :], axis=0)

        # try to load log file and compute trial averages
        try:
            # parse log file to get order of stimuli
            prot = parse_prot(prot_file_name)
            # print(prot['stimFileName'])

            # average trials by stimulus
            X = trial_traces - spd_base  # subtract baseline from each trial
            S = np.repeat(prot['stimFileName'][0:], nflies)             # grouping by STIM
            F = np.tile(list(range(nflies)), int(S.shape[0] / nflies))  # grouping by FLY
            stimnames, Sidx = np.unique(S, return_inverse=True)
            print(stimnames)
            SF = 100 * Sidx + F  # grouping by STIM and FLY

            stimfly_labels, stimfly_mean = stats_by_group(X, SF, np.nanmean)
        except Exception as e:
            print(e)
            stimfly_labels = None
            stimfly_mean = None
            stimnames = ['unknown']

        print(f'saving to {save_file_name}')
        with h5py.File(save_file_name, 'w') as f:
            f.create_dataset('spd_base', data=spd_base, compression='gzip')
            f.create_dataset('spd_test', data=spd_test, compression='gzip')
            f.create_dataset('trial_traces', data=trial_traces, compression='gzip')
            f.create_dataset('fly_ids', data=SF, compression='gzip')
            f.create_dataset('led_onsets', data=led_onsets, compression='gzip')
            f.create_dataset('led_offsets', data=led_offsets, compression='gzip')
            f.create_dataset('stimfly_labels', data=stimfly_labels, compression='gzip')
            f.create_dataset('stimfly_mean', data=stimfly_mean, compression='gzip')
            f.create_dataset('track_file_name', data=np.array([track_file_name], dtype=object), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('stim_names', data=np.array(stimnames, dtype=object), dtype=h5py.special_dtype(vlen=str))
    else:
        print('ERROR: no LED onsets found. will not save.')
