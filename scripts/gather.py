
# Read experiment database from google sheets
# uses pandas to directly fetch the sheet as a csv file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
plt.ion()
plt.style.use('seaborn-deep')


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


# load table from url
url= 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQGO1qSnt8eJDQnRTKrX7lWzUjggcabhubrwLngQbf8Ssghd243lAXItTZCjjlH7vIHgYpw3VrEz-_V/pub?gid=0&single=true&output=csv'
group_by_key = 'experiment group'
data = pd.read_csv(url)

playlists = list(set(data[group_by_key]))  # get unique analysis groups
playlists = [p for p in playlists if p is not np.nan]  # filter out nan fields
print(f'all playlists ("{group_by_key}"):')
print(playlists)

counts = [sum(playlists[ii] == data[group_by_key]) for ii in range(len(playlists))]
# plt.bar(range(len(counts)), counts)
# plt.axhline(10)
# plt.xticks(range(-1, len(counts)), playlists, rotation=60)

# now load all `*_spd.h5` files in 'res/' for this playlist
dir_res = '/Volumes/ukme04/#Common/playback/res'
dir_save = '/Volumes/ukme04/#Common/playback/tuningcurvedata'
# dir_save = '/Users/jclemens/Dropbox/playback.new/tuningcurvedata'

# get all recordings for the first (non-nan) analysis group
for playlist in ['ipituneShortSoft_NM91_male', 'ipituneShortSoft_NM91_female']:# playlists:
    print('all recordings for {}:'.format(playlist))
    all_recordings = data['filename'][playlist == data[group_by_key]]
    # print(all_recordings)

    group_labels = np.zeros((0,))
    traces = np.zeros((4000, 0))
    trial_traces = np.zeros((4000, 0))
    trial_ids = np.zeros((0,))
    rec_id = np.zeros((0,))
    stim_names = np.zeros((0,), dtype=object)
    recording_name = []
    recording_id = []
    for idx, recording in enumerate(all_recordings):
        results_file = '{0}/{1}/{1}_spd.h5'.format(dir_res, recording)
        if os.path.exists(results_file):
            print(f'   loading {results_file}')
            with h5py.File(results_file, 'r') as f:
                # aggregate all fields across results_files
                group_labels = np.concatenate((group_labels, f['stimfly_labels'][:]))
                try:
                    trial_ids = np.concatenate((trial_ids, f['fly_ids'][:]))
                    trial_traces = np.concatenate((trial_traces, f['trial_traces'][:]), axis=1)
                except:
                    pass
                traces = np.concatenate((traces, f['stimfly_mean'][:]), axis=1)
                rec_id = np.concatenate((rec_id, idx * np.ones(f['stimfly_labels'][:].shape)))
                stim_names = np.concatenate((stim_names, f['stim_names'][:]))
                recording_name.append(recording)
                recording_id.append(idx)
        else:
            print(f'   not found {results_file}')

    # print(stim_names)
    stim_id = np.floor(group_labels / 100)
    fly_id = np.mod(group_labels, 100)
    print(os.path.join(dir_save, playlist + '.h5'))
    with h5py.File(os.path.join(dir_save, playlist + '.h5'), 'w') as f:
        f.create_dataset('stim_id', data=stim_id, compression='gzip')
        f.create_dataset('rec_id', data=rec_id, compression='gzip')
        f.create_dataset('fly_id', data=fly_id, compression='gzip')
        f.create_dataset('trial_traces', data=trial_traces, compression='gzip')
        f.create_dataset('traces', data=traces, compression='gzip')
        f.create_dataset('group_labels', data=group_labels, compression='gzip')
        f.create_dataset('stim_names', data=stim_names, dtype=h5py.special_dtype(vlen=str))
        f.create_dataset('file_names', data=np.array(recording_name, dtype=object), dtype=h5py.special_dtype(vlen=str))
        f.create_dataset('recording_id', data=recording_id, compression='gzip')

    # group_labels, group_mean = stats_by_group(traces, stim_id, np.nanmean)
    # fig = plt.gcf()
    # fig.set_size_inches(40, 5)

    # plt.plot(group_mean)
    # plt.legend(group_labels)
    # plt.show()
    # print(traces.shape)
    # print(stim_id.shape)
