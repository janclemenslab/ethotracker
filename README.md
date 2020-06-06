# Ethotracker

## Installation
Install dependencies using `conda`:
```shell
conda create -n tracker python==3.7 -y
conda activate tracker
conda install numpy scipy h5py opencv pandas pyyaml matplotlib scikit-image xarray ipython
conda install deepdish peakutils -c conda-forge
python -m pip install defopt
python -m pip install git+http://github.com/postpop/videoreader --no-deps
python -m pip install git+https://github.com/postpop/attrdict --no-deps
```

Install the package for production:
```shell
python -m pip install git+https://github.com/janclemenslab/ethotracker
```
or for development:
```shell
git clone https://github.com/janclemenslab/ethotracker
cd ethotracker
python -m pip install -e .
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
