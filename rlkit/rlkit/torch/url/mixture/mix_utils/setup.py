from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("_utils.pyx"), build_dir="cython", include_dirs=[numpy.get_include()])
