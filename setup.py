from setuptools import setup, find_packages

setup(name='ethotracker',
      version='0.21',
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
