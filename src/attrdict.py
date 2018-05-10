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

    def __init__(self, d=None, default_factory=lambda: None, **kwargs):
        """Init with dict or key, name pairs."""
        super().__init__(default_factory)
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for key, value in d.items():
            setattr(self, key, value)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def save(self, filename, compression='blosc'):
        dd.io.save(filename, self, compression=compression)

    def load(self, filename, compression='blosc'):
        return AttrDict(dd.io.load(filename))
