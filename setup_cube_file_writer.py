from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Efficient cube file writer',
  ext_modules = cythonize("cube_file_writer.pyx"),
)
