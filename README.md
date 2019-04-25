# Ethotracker

## Installation
Install dependencies using `conda`:
```shell
conda install numpy scipy h5py peakutils opencv pandas pyyaml matplotlib scikit-image xarray
conda install deepdish -c conda-forge
pip install defopt
pip install git+http://github.com/postpop/videoreader
pip install git+https://github.com/postpop/attrdict
```

Install the package for production:
```shell
pip install git+https://github.com/janclemenslab/ethotracker
```
or for development:
```shell
git clone https://github.com/janclemenslab/ethotracker
cd ethotracker
pip install -e .
```

## Usage

Init tracker
```shell
python -m init...
```

Track:
```shell
python -m track...
```

Postprocess
```shell
python -m post...
```

## TODO
- [ ] implement chamber abstract class finder which can either take annotation file data or detects chambers automatically
- [ ] document the code!!
- [ ] add code for diagnostic plots (LED, all x-pos, y-pos, speeds, speed prob dists)
