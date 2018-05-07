"""A dictionary on stereoids(sic!)."""
import deepdish as dd
from collections import defaultdict


class AttrDict(defaultdict):
    """Dictionaries with dot-notation and default values and deepdish hdf5 io.

    # dictionary with default value 42 for new keys (defaults to None)
    ad = AttrDict(lambda: 42)

    # save to file with zlib compression (defaults to blosc)
    ad.save(filename, compression='zlib')

    # load from file
    ad = AttrDict().load(filename)
    """

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def save(self, filename, compression='blosc'):
        dd.io.save(filename, self, compression=compression)

    def load(self, filename, compression='blosc'):
        self = dd.io.load(filename)
