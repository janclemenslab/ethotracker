#!/usr/bin/env python3
import numpy as np
import h5py
import scipy.signal
import sys


def load_data(file_name):
    with h5py.File(file_name, 'r') as f:
        pos = f['centers'][:]
        led = f['led'][:]
        start_frame = f['start_frame'].value
    pos = pos[start_frame+1:-1000,:,:]
    led = led[start_frame+1:-1000,0].T
    nflies = pos.shape[1]
    return pos, led, nflies

def parse_prot(filename):
    with open(filename) as f:
        content = f.readlines()

    prot = {'timestamp':[], 'stimFileName':[], 'silencePre':[], 'silencePost':[], 'delayPost':[], 'intensity':[], 'freq':[], 'MODE':[]}
    for line in content:
        token = line.split(' ')
        if len(token)>1: # consider only rows that contain protocol logs
            ts, id, *data = token 
            prot['timestamp'].append(ts)
            prot_fields = data[0].split(';')
            for field in prot_fields[:-1]:
                key, value = field.split(',')
                prot[key].append(value)
    return prot

def get_led_peaks(led, thres=0.8):
    import peakutils
    from scipy import signal

    led_diff = np.diff(signal.savgol_filter(led, 11, 6)).T 
    led_onsets = peakutils.indexes(led_diff, thres=0.8)
    led_offsets = peakutils.indexes(-led_diff, thres=0.8)
    return led_onsets, led_offsets

def get_speed(pos, medfiltlen=None ):
    spd = np.sqrt(np.sum(np.diff(pos, axis=0).astype(np.float32)**2, axis=2))
    if medfiltlen is not None:
        spd = scipy.signal.medfilt(spd, medfiltlen)
    return spd

def chunk_data(data, cutpoints, chunklen):
    data_chunks = np.zeros((chunklen,10000))
    cnt = 0
    for cutpoint in cutpoints:
        this = data[cutpoint:cutpoint+chunklen,:]
        data_chunks[:, cnt:cnt+12] = this
        cnt += this.shape[1]
    data_chunks = data_chunks[:,:cnt]  # keep only non-empty traces
    return data_chunks

def stats_by_group(data, grouping, fun):
    # TODO: pass through args and kwargs for fun
    
    # trim inputs
    data = data[:,:min(data.shape[1], grouping.shape[0])]
    grouping = grouping[:min(data.shape[1], grouping.shape[0])]

    group_labels, groups = np.unique(grouping, return_inverse=True)
    # pre-allocate 
    group_stats = np.zeros((data.shape[0], len(group_labels)))
    # calc stat FUN for each group
    for idx, group in enumerate(np.unique(groups)):
        group_stats[:,idx] = fun(data[:, groups==group], axis=1)
    
    return group_labels, group_stats


if __name__ == '__main__':
    # these should be arguments
    # dir_name = '.'
    # rec_name = 'rpi8-20180129_170709'

    # track_file_name = '{0}/{1}/{1}.h5'.format(dir_name, rec_name)
    # prot_file_name  = '{0}/{1}/{1}_snd.log'.format(dir_name, rec_name)
    # save_file_name  = '{0}/{1}/{1}_spd.h5'.format(dir_name, rec_name)
    
    track_file_name = sys.argv[1]
    prot_file_name = sys.argv[2]
    save_file_name = sys.argv[3]

    # read tracking data
    pos, led, nflies = load_data(track_file_name)

    # parse log file to get order of stimuli
    prot = parse_prot(prot_file_name)
    print(prot['stimFileName'])

    led_onsets, led_offsets = get_led_peaks(led, thres=0.8)
    spd = get_speed(pos, 7)

    chunklen = 4000
    chunkpre = 2000
    trial_traces = chunk_data(spd, led_onsets[:-1] - chunkpre, chunklen)

    # calc base line and test spd
    spd_test = np.nanmean(trial_traces[2000:2400, :], axis=0)
    spd_base = np.nanmean(trial_traces[1000:1800, :], axis=0)

    # average trials by stimulus
    X = trial_traces - spd_base  # subtract baseline from each trial
    S = np.repeat(prot['stimFileName'][0:], nflies)             # grouping by STIM
    F = np.tile(list(range(nflies)), int(S.shape[0] / nflies))  # grouping by FLY
    stimnames, Sidx = np.unique(S, return_inverse=True)
    print(stimnames)
    SF = 100*Sidx + F  # grouping by STIM and FLY

    stimfly_labels, stimfly_mean = stats_by_group(X, SF, np.nanmean)

    with h5py.File(save_file_name, 'w') as f:
        # f.create_dataset('stimnames', data=stimnames)
        f.create_dataset('stimfly_labels', data=stimfly_labels, compression='gzip')
        f.create_dataset('stimfly_mean', data=stimfly_mean, compression='gzip')
        f.create_dataset('spd_base', data=spd_base, compression='gzip')
        f.create_dataset('spd_test', data=spd_test, compression='gzip')
        # f.create_dataset('track_file_name', value=file_name, dtype=h5py.special_dtype(vlen=unicode))
