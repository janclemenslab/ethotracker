"""Functions for correcting tracks - currently implemented: fix head-tail orientation."""
import numpy as np
import flammkuchen
import scipy.signal
import logging
import defopt

def smooth(x, N):
    """Smooth signal using box filter of length N samples."""
    return np.convolve(x, np.ones((N,)) / N, mode='full')[(N - 1):]


def fix_orientations(lines0, chamber_number=0):
    """Fix the head-tail orientation of flies based on speed and changes in orientation."""
    nflies = lines0.shape[2]
    vel = np.zeros((lines0.shape[0], nflies))
    ori = np.zeros((lines0.shape[0], 2, nflies))
    lines_fixed = lines0.copy()
    chamber_number = 0

    for fly in range(nflies):
        # get fly lines and smooth
        lines = lines0[:, chamber_number, fly, :, :].astype(np.float64)  # time x [head,tail] x [x,y]
        for h in range(2):
            for p in range(2):
                lines[:, h, p] = smooth(lines[:, h, p], 10)

        # get fly movement and smooth
        dpos = np.gradient(lines[:, 0, :], axis=0)  # change of head position over time - `np.gradient` is less noisy than `np.diff`
        for p in range(2):
            dpos[:, p] = smooth(dpos[:, p], 10)

        # get fly orientation
        ori[:, :, fly] = np.diff(lines, axis=1)[:, 0, :]  # orientation of fly: head pos - tail pos, `[:,0,:]` cuts off trailing dim
        ori_norm = np.linalg.norm(ori[:, :, fly], axis=1)  # "length" of the fly

        # dpos_norm = np.linalg.norm(dpos, axis=1)
        # alignment (dot product) between movement and orientation
        dpos_ori = np.einsum('ij,ij->i', ori[:, :, fly], dpos)  # "element-wise" dot product between orientation and velocity vectors
        vel[:, fly] = dpos_ori / ori_norm  # normalize by fly length (norm of ori (head->tail) vector)

        # 1. clean up velocity - only consider epochs were movement is fast and over a prolonged time
        orichange = np.diff(np.unwrap(np.arctan2(ori[:, 0, fly], ori[:, 1, fly])))  # change in orientation for change point detection
        velsmooththres = smooth(vel[:, fly], 20)  # smooth velocity
        velsmooththres[np.abs(velsmooththres) < 0.4] = 0  # threshold - keep only "fast" events to be more robust
        velsmooththres = scipy.signal.medfilt(velsmooththres, 5)  # median filter to get rid of weird, fast spikes in vel

        # 2. detect the range of points during which velocity changes in sign
        idx, = np.where(velsmooththres != 0)  # indices where vel is high
        switchpoint = np.gradient(np.sign(velsmooththres[idx]))  # changes in the sign of the thres vel
        switchtimes = idx[np.where(switchpoint != 0)]  # indices where changes on vel sign occurr
        switchpoint = switchpoint[switchpoint != 0]    # sign of the change in vel

        # 3. detect actual change point with that range
        changepoints = []  # fill this with
        for cnt in range(0, len(switchtimes[:-1]), 2):
            # define change points as maxima in orientation changes between switchs in direction of motion
            changepoints.append(switchtimes[cnt] + np.argmax(np.abs(orichange[switchtimes[cnt] - 1:switchtimes[cnt + 1]])))
            # mark change points for interpolation
            velsmooththres[changepoints[-1]-1] = -switchpoint[cnt]
            velsmooththres[changepoints[-1]] = switchpoint[cnt+1]

        # 4. fill values using change points - `-1` means we need to swap head and tail
        idx, = np.where(velsmooththres != 0)  # update `idx` to include newly marked change points
        f = scipy.interpolate.interp1d(idx, velsmooththres[idx], kind="nearest", fill_value="extrapolate")
        # idx_new, = np.where(velsmooththres == 0)
        ynew = f(range(velsmooththres.shape[0]))

        # 4. swap head and tail
        lines_fixed[ynew < 0, chamber_number, fly, :, :] = lines_fixed[ynew < 0, chamber_number, fly, ::-1, :]
    return lines_fixed


def run(track_file_name: str, save_file_name: str):
    """Load data, call fix_orientations and save data."""
    logging.info(f"   processing tracks in {track_file_name}. will save to {save_file_name}")
    data = flammkuchen.load(track_file_name)
    data['lines'] = fix_orientations(data['lines'])
    logging.info(f"   saving fixed tracks to {save_file_name}")
    flammkuchen.save(save_file_name, data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    defopt.run(run)
