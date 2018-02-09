import h5py


class Results():
    def __init__(self, background=None, chambers=None, centers=None, lines=None, led=None,
                 frame_count=None, nflies=None, file_name=None, status=None, start_frame=None,
                 chambers_bounding_box=None):
        # initialize all fields
        self.background = background
        self.chambers = chambers
        self.chambers_bounding_box = chambers_bounding_box
        self.centers = centers
        self.lines = lines
        self.led = led
        self.frame_count = frame_count
        self.start_frame = start_frame
        self.nflies = nflies
        self.status = status
        # load file
        if file_name is not None:
            self.load(file_name)

    def save(self, file_name):
        with h5py.File(file_name, 'w') as h5f:
            for key, value in self.__dict__.items():
                if value is not None:
                    try:  # try with compression
                        h5f.create_dataset(key, data=value, compression='gzip')
                    except:  # if fails (for scalars, strings et al.) try w/o
                        h5f.create_dataset(key, data=value)
                else:
                    h5f.create_dataset(key, data='None')  # save None values as empty data sets

    def load(self, file_name):
        with h5py.File(file_name, 'r') as h5f:
            for key, item in h5f.items():
                setattr(self, key, item.value)
                if str(item.value) in 'None':
                    setattr(self, key, None)  # read empty data sets as None values


def test():
    print("empty")

if __name__ == "__main__":
    test()
