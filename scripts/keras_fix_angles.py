"""Manual annotation of videos"""
import numpy as np
from videoreader import VideoReader
import tracker.foreground as fg
import h5py
import time
import sys


def load_model(file_name):
    from keras.models import model_from_json
    model = model_from_json(open(file_name + '_arch.json').read())
    d = np.load(file_name + '_weights.npz')
    model.set_weights(d['weights'])
    model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
    return model


def vid_orientation(vidfile, trackfile, modelfile, savefile, flybox_width=25):
    """Fix angles."""

    model = load_model(modelfile)
    print(f'\n Loaded model version: {modelfile}')

    vr = VideoReader(vidfile)
    print(f'\n Loading video file: {vidfile}')

    # load tracker results
    with h5py.File(trackfile, 'r') as f:
        centers = f['centers'][:]
        chbb = f['chambers_bounding_box'][:]
    nflies = centers.shape[2]
    start_frame = 1001
    chamber_number = 0
    nframes = centers.shape[0]
    frame_offset = 0
    pred_chunk = 1000        # how many frames should be predicted together?

    #%% read frames
    pred = np.zeros((nframes,nflies))
    start_time = time.time()

    print(f'Video analysis starting at frame {start_frame}:')

    for frame_number in range(start_frame,nframes):

        raw_frame = vr[frame_number]
        if raw_frame is not None:
            # get the current frame
            frame = fg.crop(raw_frame, np.ravel(chbb[chamber_number+1][:, ::-1]))

            # get boxes for all flies in this frame
            for fly_number in range(nflies):
                pos = centers[int(frame_number) + frame_offset, 0, fly_number, :]
                flybox = frame[int(pos[0] - flybox_width):int(pos[0] + flybox_width),
                               int(pos[1] - flybox_width):int(pos[1] + flybox_width), 0]
                flybox = flybox[np.newaxis,:,:,np.newaxis]
                pred[frame_number,fly_number]=model.predict(flybox)

            # give update after pred_chunk frames
            if frame_number%pred_chunk==0:
                progress = str(frame_number/nframes)[:4]
                curr_time = time.time()
                duration = str(curr_time - start_time)[:5]
                print('Made predictions for frames',frame_number-pred_chunk+1,'to',frame_number+1,'of',nframes,f'({progress} %)   ({duration}s)')
                start_time = time.time()

    # save results
    print(f'\n Saving to {savefile}.\n')
    with h5py.File(savefile, 'w') as f:
        dset = f.create_dataset('pred', data=pred, compression='gzip')
        dset.attrs['model_version'] = str(modelfile)

    #%% call function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Uses keras network model to fix fly angles.')
    parser.add_argument('videofile', type=str, help='video file')
    parser.add_argument('trackfile', type=str, help='file with tracks')
    parser.add_argument('modelfile', type=str, help='file containing the keras network model')
    parser.add_argument('savefile', type=str, help='file to save fixed angles to')
    args = parser.parse_args()

    # ori_tracker_cluster videofile trackfile modelfile savefile
    vid_orientation(args.videofile, args.trackfile, args.modelfile, args.savefile)
    print('all done!\n \n')
