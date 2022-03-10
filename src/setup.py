#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup compilation of Cythpon-files.

Author:
    Erik Johannes Husom

Created:
    2022-03-10 torsdag 13:41:27 

To compile, run:

    python3 src/setup.py build_ext --inplace

"""

import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(name="Cython modules",
        ext_modules=cythonize("src/cutils.pyx", 
            compiler_directives={"language_level": "3"}),
        include_dirs=[numpy.get_include()])

