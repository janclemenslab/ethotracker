#!/usr/bin/env python3
import numpy as np
import h5py
import peakutils
import os
import scipy.signal
import argparse
from post.networkmotifs import process_motifs
from post.chainingindex import get_chainlength, get_chaining
from post.fixtracks import fix_orientations


def load_data(file_name):
    with h5py.File(file_name, 'r') as f:
        pos = f['centers'][:]
        led = f['led'][:]
        lines = f['lines'][:]
        start_frame = f['start_frame'].value
    pos = pos[start_frame + 1:-1000, :, :]
    led = led[start_frame + 1:-1000, 0].T

    nflies = pos.shape[2]
    return pos, lines, led, nflies


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
    led_diff = np.diff(scipy.signal.savgol_filter(led, 11, 6)).T
    led_onsets = peakutils.indexes(led_diff, thres=thres)
    led_offsets = peakutils.indexes(-led_diff, thres=thres)

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

    # plt.show()
    # plt.axis('tight')
    if savefilename:
        print(savefilename)
        plt.savefig(savefilename)


def get_speed(pos, medfiltlen=None):
    spd = np.sqrt(np.sum(np.diff(pos, axis=0).astype(np.float32)**2, axis=2))
    if medfiltlen is not None:
        for fly in range(spd.shape[1]):
            spd[:, fly] = scipy.signal.medfilt(spd[:, fly], medfiltlen)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('track_file_name', type=str, help='track file to process')
    parser.add_argument('save_file_name', type=str, help='file to save results')
    parser.add_argument('-p', '--prot_file_name', type=str, help='protocol file')
    parser.add_argument('--networkmotifs', action='store_true', help='find network motifs (SLOW!)')
    args = parser.parse_args()

    print('processing tracks in {0}. will save to {1}'.format(args.track_file_name, args.save_file_name))
    chamber_number = 0
    # read tracking data
    pos, lines, led, nflies = load_data(args.track_file_name)
    # fix lines and get chaining IndexError
    lines_fixed = fix_orientations(lines)
    chainee, chainer, headee, header, D_h2t, D_h2h, Dc, Dh = get_chaining(lines_fixed, chamber_number)
    chain_length = get_chainlength(chainer, chainee, nflies)
    # save fixed lines and chaining data_chunks
    print(f'saving chaining data to {args.save_file_name}')
    with h5py.File(args.save_file_name, 'w') as f:
        f.create_dataset('lines_fixed', data=lines_fixed, compression='gzip')
        f.create_dataset('chain_length', data=chain_length, compression='gzip')
        f.create_dataset('chainer', data=chainer, compression='gzip')
        f.create_dataset('chainee', data=chainee, compression='gzip')
        f.create_dataset('header', data=header, compression='gzip')
        f.create_dataset('headee', data=headee, compression='gzip')
        f.create_dataset('D_h2t', data=D_h2t, compression='gzip')
        f.create_dataset('D_h2h', data=D_h2h, compression='gzip')
        f.create_dataset('Dc', data=Dc, compression='gzip')
        f.create_dataset('Dh', data=Dh, compression='gzip')
        f.create_dataset('track_file_name', data=np.array([args.track_file_name], dtype=object), dtype=h5py.special_dtype(vlen=str))

    # detect LED onsets
    led_onsets, led_offsets = get_led_peaks(led[0], thres=0.8, min_interval=1000)
    print('found {0} led onsets'.format(len(led_onsets)))
    print(f'saving led data')
    with h5py.File(args.save_file_name, 'a') as f:
        f.create_dataset('led_onsets', data=led_onsets, compression='gzip')
        f.create_dataset('led_offsets', data=led_offsets, compression='gzip')

    if led_onsets:
        try:
            spd = get_speed(pos[:, chamber_number, :, :])
            # chunk data
            chunklen = 13000
            chunkpre = 3000
            good_idx = (led_onsets - chunkpre + chunklen) <= spd.shape[0]  # ensure we don't exceed bounds - ignore too late LEDs
            trial_traces = chunk_data(spd, led_onsets[good_idx] - chunkpre, chunklen)
            # calc base line and test spd
            spd_test = np.nanmean(trial_traces[3000:3400, :], axis=0)
            spd_base = np.nanmean(trial_traces[1000:1800, :], axis=0)
            print('saving speed data')
            with h5py.File(args.save_file_name, 'a') as f:
                f.create_dataset('spd_base', data=spd_base, compression='gzip')
                f.create_dataset('spd_test', data=spd_test, compression='gzip')
                f.create_dataset('trial_traces', data=trial_traces, compression='gzip')

            # try to load log file and compute trial averages
            if args.protocol_file_name:
                # parse log file to get order of stimuli
                prot = parse_prot(args.prot_file_name)
                print(prot['stimFileName'])

                # average trials by stimulus
                X = trial_traces - spd_base  # subtract baseline from each trial
                S = np.repeat(prot['stimFileName'][0:], nflies)             # grouping by STIM
                F = np.tile(list(range(nflies)), int(S.shape[0] / nflies))  # grouping by FLY
                stimnames, Sidx = np.unique(S, return_inverse=True)
                print(stimnames)
                SF = 100 * Sidx + F  # grouping by STIM and FLY

                stimfly_labels, stimfly_mean = stats_by_group(X, SF, np.nanmean)
            else:
                # fall back values
                stimfly_labels = np.zeros((0, 0))
                stimfly_mean = np.zeros((0, 0))
                stimnames = ['unknown']

            print(f'saving tuning data to {args.save_file_name}')
            with h5py.File(args.save_file_name, 'a') as f:
                f.create_dataset('stimfly_labels', data=stimfly_labels, compression='gzip')
                f.create_dataset('stimfly_mean', data=stimfly_mean, compression='gzip')
                f.create_dataset('stim_names', data=np.array(stimnames, dtype=object), dtype=h5py.special_dtype(vlen=str))
        except Exception as e:
            print(e)

    # network motifs
    if args.networkmotifs:
        print('   analyzing network motifs')
        motif_counts, motifs = process_motifs(chainer, chainee, max_k=4)
        print(f'saving network motif results')
        with h5py.File(args.save_file_name, 'a') as f:
            f.create_dataset('motif_counts', data=motif_counts, compression='gzip')
            f.create_dataset('motifs', data=motifs, compression='gzip')
