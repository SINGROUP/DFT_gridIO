from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Locpot import',
  ext_modules = cythonize("locpot_import.pyx"),
)
