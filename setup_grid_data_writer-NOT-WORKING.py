from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Grid data writer',
  ext_modules = cythonize("grid_data_writer.pyx"),
)
