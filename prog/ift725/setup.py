# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension('im2col_cython', ['im2col_cython.pyx'],
              include_dirs=[numpy.get_include()]
              ),
]

setup(
    ext_modules=cythonize(extensions),
)
