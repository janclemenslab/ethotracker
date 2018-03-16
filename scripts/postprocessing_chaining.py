#!/usr/bin/env python3
import numpy as np
import h5py
import scipy
import sys
import peakutils
from scipy import signal
import os


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
    led_diff = np.diff(signal.savgol_filter(led, 11, 6)).T
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


def smooth(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='full')[(N-1):]


def fix_orientations(lines0, chamber_number=0):
    # fix orientation direction of flies
    nflies = lines0.shape[3]
    vel = np.zeros((lines0.shape[0], nflies))
    ori = np.zeros((lines0.shape[0], 2, nflies))
    lines_fixed = lines0.copy()

    for fly in range(nflies):
        # get fly lines and smooth
        lines = lines0[:,chamber_number, fly,:,:].astype(np.float64)  # time x [head,tail] x [x,y]
        for h in range(2):
            for p in range(2):
                lines[:,h,p] = smooth(lines[:,h,p], 10)

        # get fly movement and smooth
        dpos = np.gradient(lines[:,0,:], axis=0)  # change of head position over time - `np.gradient` is less noisy than `np.diff`
        for p in range(2):
            dpos[:,p] = smooth(dpos[:,p], 10)

        # get fly orientation
        ori[:,:,fly] = np.diff(lines, axis=1)[:,0,:]  # orientation of fly: head pos - tail pos, `[:,0,:]` cuts off trailing dim
        ori_norm = np.linalg.norm(ori[:,:,fly], axis=1)  # "length" of the fly

        # dpos_norm = np.linalg.norm(dpos, axis=1)
        # alignment (dot product) between movement and orientation
        dpos_ori = np.einsum('ij,ij->i', ori[:,:,fly], dpos)  #"element-wise" dot product between orientation and velocity vectors
        vel[:, fly] = dpos_ori/ori_norm  # normalize by fly length (norm of ori (head->tail) vector)

        # 1. clean up velocity - only consider epochs were movement is fast and over a prolonged time
        orichange = np.diff(np.unwrap(np.arctan2(ori[:,0,fly], ori[:,1,fly])))  # change in orientation for change point detection
        velsmooththres = smooth(vel[:,fly],20)  # smooth velocity
        velsmooththres[np.abs(velsmooththres)<0.4] = 0  # threshold - keep only "fast" events to be more robust
        velsmooththres = scipy.signal.medfilt(velsmooththres, 5)  # median filter to get rid of weird, fast spikes in vel

        # 2. detect the range of points during which velocity changes in sign
        idx, = np.where(velsmooththres!=0)  # indices where vel is high
        switchpoint = np.gradient(np.sign(velsmooththres[idx]))  # changes in the sign of the thres vel
        switchtimes = idx[np.where(switchpoint!=0)]  # indices where changes on vel sign occurr
        switchpoint = switchpoint[switchpoint!=0]    # sign of the change in vel

        # 3. detect actual change point with that range
        changepoints = []  # fill this with
        for cnt in range(0, len(switchtimes[:-1]),2):
            # define change points as maxima in orientation changes between switchs in direction of motion
            changepoints.append(switchtimes[cnt]+np.argmax(np.abs(orichange[switchtimes[cnt]-1:switchtimes[cnt+1]])))
            # mark change points for interpolation
            velsmooththres[changepoints[-1]-1] = -switchpoint[cnt]
            velsmooththres[changepoints[-1]] = switchpoint[cnt+1]

        # 4. fill values using change points - `-1` means we need to swap head and tail
        idx, = np.where(velsmooththres!=0)  # update `idx` to include newly marked change points
        f = scipy.interpolate.interp1d(idx, velsmooththres[idx], kind="nearest", fill_value="extrapolate");
        idx_new, = np.where(velsmooththres==0)
        ynew = f(range(velsmooththres.shape[0]))

        # 4. swap head and tail
        lines_fixed[ynew<0,chamber_number,fly,:,:] = lines_fixed[ynew<0,chamber_number,fly,::-1,:]
    return lines_fixed


def get_chaining(lines, chamber_number=0):
    nframes = lines.shape[0]
    nflies = lines.shape[2]
    tails = lines[:,chamber_number,:,0,:]
    heads = lines[:,chamber_number,:,1,:]

    D_h2t = np.zeros((nflies, nflies, nframes))
    D_h2h = np.zeros((nflies, nflies, nframes))
    Dc = np.zeros((nflies, nflies, nframes), dtype=np.bool)
    Dh = np.zeros((nflies, nflies, nframes), dtype=np.bool)

    chainee = -np.ones((nflies*nflies, nframes), dtype=np.int16)
    chainer = -np.ones((nflies*nflies, nframes), dtype=np.int16)
    headee = -np.ones((nflies*nflies, nframes), dtype=np.int16)
    header = -np.ones((nflies*nflies, nframes), dtype=np.int16)

    for frame_number in range(0,nframes):
        T = frame_number
        D_h2t[:,:,T] = scipy.spatial.distance.cdist(tails[T,:,:], heads[T,:,:], metric='euclidean')
        D_h2h[:,:,T] = scipy.spatial.distance.cdist(heads[T,:,:], heads[T,:,:], metric='euclidean')

        flylength = np.diag(D_h2t[:,:,T])  # diagonals contain tail->head distances=fly lengths
        min_distance = np.min(flylength)  # interaction distance is flylength+x
        Dc[:,:,T] = D_h2t[:,:,T]<min_distance   # chaining
        Dh[:,:,T] = D_h2h[:,:,T]<min_distance/2 # head-butting
        # ignore diagonal entries
        np.fill_diagonal(Dc[:,:,T], False)
        np.fill_diagonal(Dh[:,:,T], False)

        # get x,y coords of interacting flies
        chainee_this, chainer_this = np.where(Dc[:,:,T])
        headee_this, header_this = np.where(Dh[:,:,T])

        # save all in list
        chainee[0:chainee_this.shape[0],T] = chainee_this
        chainer[0:chainer_this.shape[0],T] = chainer_this
        headee[0:headee_this.shape[0],T] = headee_this
        header[0:header_this.shape[0],T] = header_this
    return chainee, chainer, headee, header, D_h2t, D_h2h, Dc, Dh


def get_chainlength(chainer, chainee, nflies):
    # calculate chain length
    # TODO: [-] this will ignore circles since there is no seed - a fly that chains but is not a chainee
    #       [x] very slow... - will run endlessly if there is a loop
    nframes = chainer.shape[1]

    chain_length = np.zeros((nflies*2, nframes), dtype=np.uint16)
    for frame_number in range(0, nframes):
        # if frame_number % 10000==0:
        #     print(frame_number)
        # find starters of chain - flies that are chainer but not chainees
        chain_seeds = [x for x in chainer[:, frame_number] if not x in chainee[:, frame_number]]

        # for each starter - get chain
        chain = [-1]*len(chain_seeds)
        for chain_count, chain_seed in enumerate(chain_seeds):
            chain[chain_count] = [chain_seed]
            chain_link = [chain_seed]
            idx = 1
            while len(chain_link):
                idx, = np.where(chainer[:, frame_number] == chain_link)
                chain_link = chainee[idx, frame_number].tolist()
                try:
                    if chain_link[0] not in chain[chain_count]:
                        chain[chain_count].append(chain_link[0])
                    else:
                        break  # avoid loop where chainee chains chainer
                except IndexError as e:
                    pass
        # find chainers that are not accounted for - these may be part of a circle...
        flat_list = [item for ch in chain for item in ch[:-1]]  # flatten list excluding last in chain
        leftovers = [x for x in flat_list if not x in chainer[:, frame_number]]  # count
        if leftovers:
            chain.append(leftovers)

        this_chain_len = [len(x) for x in chain]
        chain_length[:len(this_chain_len), frame_number] = this_chain_len
    #     print(f"found {len(chain_seeds)} chains with {this_chain_len} flies: {chain}")
    return chain_length


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
    pos, lines, led, nflies = load_data(track_file_name)
    # fix lines and get chaining IndexError
    lines_fixed = fix_orientations(lines)
    chainee, chainer, headee, header, D_h2t, D_h2h, Dc, Dh = get_chaining(lines_fixed, chamber_number=0)
    chain_length = get_chainlength(chainer, chainee, nflies)
    # save fixed lines and chaining data_chunks
    print(f'saving chaining data to {save_file_name}')
    with h5py.File(save_file_name, 'w') as f:
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

    # detect LED onsets
    led_onsets, led_offsets = get_led_peaks(led[0], thres=0.8, min_interval=1000)
    print(led_onsets)
    # plot LEDs and save fig
    try:
        plot_led_peaks(led[0], led_onsets, led_offsets, os.path.splitext(save_file_name)[0]+'.png')
    except Exception as e:
        pass

    if len(led_onsets):
        print('found {0} led onsets'.format(len(led_onsets)))
        spd = get_speed(pos[:, 0, :, :], 7)
        # chunk data
        chunklen = 13000
        chunkpre = 3000
        good_idx = (led_onsets - chunkpre + chunklen) <= spd.shape[0]  # ensure we don't exceed bounds - ignore too late LEDs
        trial_traces = chunk_data(spd, led_onsets[good_idx] - chunkpre, chunklen)
        # calc base line and test spd
        spd_test = np.nanmean(trial_traces[3000:3400, :], axis=0)
        spd_base = np.nanmean(trial_traces[1000:1800, :], axis=0)

        # try to load log file and compute trial averages
        try:
            # parse log file to get order of stimuli
            prot = parse_prot(prot_file_name)
            print(prot['stimFileName'])

            # average trials by stimulus
            X = trial_traces - spd_base  # subtract baseline from each trial
            S = np.repeat(prot['stimFileName'][0:], nflies)             # grouping by STIM
            F = np.tile(list(range(nflies)), int(S.shape[0] / nflies))  # grouping by FLY
            stimnames, Sidx = np.unique(S, return_inverse=True)
            print(stimnames)
            SF = 100 * Sidx + F  # grouping by STIM and FLY

            stimfly_labels, stimfly_mean = stats_by_group(X, SF, np.nanmean)
        except Exception as e:
            print('error while reading snd_log and stim averaging:')
            print(e)
            # fall back values
            stimfly_labels = np.zeros((0, 0))
            stimfly_mean = np.zeros((0, 0))
            stimnames = ['unknown']

        print(f'saving to {save_file_name}')
        with h5py.File(save_file_name, 'a') as f:
            f.create_dataset('spd_base', data=spd_base, compression='gzip')
            f.create_dataset('spd_test', data=spd_test, compression='gzip')
            f.create_dataset('trial_traces', data=trial_traces, compression='gzip')
            f.create_dataset('led_onsets', data=led_onsets, compression='gzip')
            f.create_dataset('led_offsets', data=led_offsets, compression='gzip')
            f.create_dataset('stimfly_labels', data=stimfly_labels, compression='gzip')
            f.create_dataset('stimfly_mean', data=stimfly_mean, compression='gzip')
            f.create_dataset('track_file_name', data=np.array([track_file_name], dtype=object), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('stim_names', data=np.array(stimnames, dtype=object), dtype=h5py.special_dtype(vlen=str))
    else:
        print('ERROR: no LED onsets found. will not save.')
