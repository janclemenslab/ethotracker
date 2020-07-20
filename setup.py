from setuptools import setup, find_packages
import codecs
import re
import os

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")
setup(name='ethotracker',
      version=find_version("src/ethotracker/__init__.py"),
      description='simple tracker',
      url='http://github.com/janclemenslab/ethotracker',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['numpy','scipy', 'h5py', 'flammkuchen', 'pandas',
                        'opencv-python', 'defopt', 'pyyaml', 'matplotlib', 'scikit-image',
                        'PeakUtils', 'pyvideoreader', 'scikit-learn'],
      include_package_data=True,
      zip_safe=False,
      )
