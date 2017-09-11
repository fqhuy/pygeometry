# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:50:28 2013

@author: phan
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = "pygeometry",
    ext_modules = cythonize('*.pyx'),
    include_dirs=[np.get_include()]
)
