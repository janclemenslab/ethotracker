# FlyPursuit - fast and simple fly tracker
## installation (windows)
```
conda install numpy scipy accelerate h5py
conda install -c menpo opencv3
git clone https://github.com/postpop/FlyPursuit.git
```

## usage
check the source code and `FlyPursuit.py --help`, `pursuit_caller --help`

## call matlab from python
installation:
```
cd "MATLABROOT/extern/engines/python"
python setup.py install
```
usage:
```python
import matlab.engine
eng = matlab.engine.start_matlab()      # start engine
eng.cd('pathToResults')                 # cd into right dir
eng.video_postProcess_python(nargout=0) # process
eng.quit()                              # quit engine
```
