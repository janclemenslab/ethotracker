
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
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRCb0k0qztnvXZqn9iHsWhKBzceZJWq5Iz4MkGvpe0rHRNR5oTVhOLQ_9robQDE0Njh8G-UqTrbPxb9/pub?output=csv'
data = pd.read_csv(url)

playlists = list(set(data['analysis group']))  # get unique analysis groups
playlists = [p for p in playlists if p is not np.nan]  # filter out nan fields
print('all playlists (analysis groups):')
print(playlists)

counts = [sum(playlists[ii]==data['analysis group']) for ii in range(len(playlists))]
plt.bar(range(len(counts)), counts)
plt.axhline(10)
plt.xticks(range(-1,len(counts)), playlists, rotation=60);

# now load all `*_spd.h5` files in 'res/' for this playlist
dir_res = '/Volumes/ukme04/#Common/playback/res'

# get all recordings for the first (non-nan) analysis group
for playlist in playlists:
    print('all recordings for {}:'.format(playlist))
    all_recordings = data['filename'][playlist==data['analysis group']]
    print(all_recordings)


    group_labels = np.zeros((0,))
    traces = np.zeros((4000,0))
    rec_id = np.zeros((0,))
    stim_names = np.zeros((0,), dtype=object)
    for idx, recording in enumerate(all_recordings):
        results_file = '{0}/{1}_spd.h5'.format(dir_res, recording)
    #     print(results_file)
        if os.path.exists(results_file):
            with h5py.File(results_file, 'r') as f:
                # aggregate all fields across results_files
                group_labels = np.concatenate((group_labels, f['stimfly_labels'][:]))
                traces = np.concatenate((traces, f['stimfly_mean'][:]), axis=1)
                rec_id = np.concatenate((rec_id, idx*np.ones(f['stimfly_labels'][:].shape)))
                stim_names = np.concatenate((stim_names, f['stim_names'][:]))
                pass
    print(stim_names)
    stim_id = np.floor(group_labels/100)
    fly_id = np.mod(group_labels,100)
    with h5py.File(playlist + '.h5','w') as f:
        f.create_dataset('stim_id', data=stim_id, compression='gzip')
        f.create_dataset('rec_id', data=rec_id, compression='gzip')
        f.create_dataset('fly_id', data=fly_id, compression='gzip')
        f.create_dataset('traces', data=traces, compression='gzip')
        f.create_dataset('group_labels', data=group_labels, compression='gzip')
        f.create_dataset('stim_names', data=stim_names, dtype=h5py.special_dtype(vlen=str))

    # group_labels, group_mean = stats_by_group(traces, stim_id, np.nanmean)
    # fig = plt.gcf()
    # fig.set_size_inches(40, 5)

    # plt.plot(group_mean)
    # plt.legend(group_labels)
    # plt.show()
    # print(traces.shape)
    # print(stim_id.shape)
