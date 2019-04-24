#!/usr/bin/env python3
import numpy as np
import h5py
import scipy.signal
import sys
import peakutils
from scipy import signal
import os
import logging
import pandas as pd

def load_data(file_name):
    with h5py.File(file_name, 'r') as f:
        if any([attr.startswith('DEEPDISH') for attr in list(f.attrs)]):
            logging.info("deepdish file.")
        else:
            pos = f['centers'][:]
            led = f['led'][:]
            start_frame = f['start_frame'].value
            end_frame = f['frame_count'].value

    import deepdish as dd
    data = dd.io.load(file_name)
    pos = data['centers']
    led = data['led']
    start_frame = data['start_frame']
    end_frame = data['frame_count']

    pos = pos[start_frame + 1:end_frame, :, :]
    led = led[start_frame + 1:end_frame, 0].T
    nflies = pos.shape[1]
    return pos, led, nflies


def parse_session_log(logfile_name):
    """Reconstructs playlist from log file.

    Args:
        logfilename
    Returns:
        dict with playlist entries
    """

    with open(logfile_name, 'r') as f:
        logs = f.read()
    log_lines = logs.strip().split('\n')
    session_log = []
    for current_line in log_lines:
        head, _, dict_str = current_line.partition(': ')

        # process head
        #     timestamp, hostname = head.split(' ')
        #     timestamp = datetime.strptime(timestamp, '%Y-%m-%d,%H:%M:%S.%f')

        if dict_str[:4] == 'cnt:':
            dict_items = dict_str.strip().split('; ')
            dd = dict()
            for dict_item in dict_items:
                key, val = dict_item.strip(';').split(': ')
                try:
                    dd[key.strip()] = eval(val.strip())
                except (ValueError, NameError):
                    dd[key.strip()] = val.strip()
            session_log.append(dd)
    return session_log


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
        spd = scipy.signal.medfilt(spd, (medfiltlen, 1))
    return spd


def chunk_data(data, cutpoints, chunklen):
    data_chunks = np.zeros((chunklen, 10000))
    cnt = 0
    for cutpoint in cutpoints:
        try:
            this = data[cutpoint:cutpoint + chunklen, :]
            data_chunks[:, cnt:cnt + data.shape[1]] = this
        except ValueError:
            data_chunks[:, cnt:cnt + data.shape[1]] = np.nan
        finally:
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
        spd = get_speed(pos[:, :, 0, :], 7)
        # chunk data
        chunklen = 4000
        chunkpre = 2000

        trial_traces = chunk_data(spd, led_onsets[:-1] - chunkpre, chunklen)
        # calc base line and test spd
        spd_test = np.nanmean(trial_traces[2000:2400, :], axis=0)
        spd_base = np.nanmean(trial_traces[1000:1800, :], axis=0)
        SF = [0]
        # try to load log file and compute trial averages
        try:
            # parse log file to get order of stimuli
            prot = pd.DataFrame(parse_session_log(prot_file_name))

            # average trials by stimulus
            X = trial_traces - spd_base  # subtract baseline from each trial
            stimfilenames_merged = [' + '.join(sf) for sf in prot['stimFileName']]
            S = np.repeat(stimfilenames_merged, nflies)             # grouping by STIM
            F = np.tile(list(range(nflies)), int(S.shape[0] / nflies))  # grouping by FLY
            stimnames, Sidx = np.unique(S, return_inverse=True)
            print(stimnames)
            SF = 100 * Sidx + F  # grouping by STIM and FLY

            stimfly_labels, stimfly_mean = stats_by_group(X, SF, np.nanmean)
        except Exception as e:
            # import ipdb; ipdb.set_trace()
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
