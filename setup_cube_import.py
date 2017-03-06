from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Cube file importer',
  ext_modules = cythonize("cube_import.pyx"),
)
