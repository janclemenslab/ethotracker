from attrdict import AttrDict


def test_none():
    ad = AttrDict()
    assert(ad.NonExistingKey == None)


def test_initdict():
    ad = AttrDict({'test': 42})
    assert ad['test'] == 42


def test_setgetkeyval():
    ad = AttrDict()
    ad['test'] = 100
    assert ad['test'] == 100


def test_setgetatttr():
    ad = AttrDict()
    ad.test = 100
    assert ad.test == 100


def test_save():
    ad = AttrDict()
    ad['test'] = list(range(100))
    ad.save('test.h5', compression='blosc')


def test_load():
    ad = AttrDict()
    ad['test'] = list(range(100))
    ad.save('test.h5', compression='blosc')
    adl = AttrDict().load('test.h5')
    assert isinstance(adl, AttrDict)
