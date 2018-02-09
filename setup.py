from setuptools import setup, find_packages

setup(name='analysis',
      version='0.10',
      description='playback analyses',
      url='http://github.com/janclemenslab',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages(exclude='jobs'),
      # install_requires=['numpy', 'scipy', 'h5py', 'peakutils', 'opencv', 'pandas'],
      include_package_data=True,
      zip_safe=False,
      # data_files=[('~/bin', ['scripts/*'])], # copy everything in scripts to ~/bin
      )
