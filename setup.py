from setuptools import setup, find_packages

setup(name='ethotracker',
      version='0.20',
      description='simple tracker',
      url='http://github.com/janclemenslab/ethotracker',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      zip_safe=False,
      )
