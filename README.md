# Ethotracker

## Installation
```shell
conda create -n tracker python==3.7 -y
conda activate tracker
python -m pip install git+https://github.com/postpop/attrdict
python -m pip install git+https://github.com/janclemenslab/ethotracker
```

## Usage

Track:
```shell
python -m ethotracker.tracker.pursuit
```

Postprocess
```shell
python -m ethotracker.post.postprocessing
```
