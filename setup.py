#! /usr/bin/env python
# coding: utf-8

from distutils.core import setup
from Cython.Build import cythonize

files = ["data/*", '*.pyx']

from distutils.core import setup
from distutils.extension import Extension

import numpy


# build extension with CYTHON?
USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'
extensions = [Extension("omi.cgrate", ["omi/cgrate"+ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name                = 'omi',
    version             = '0.1',
    description         = 'gridding algorithm for OMI data',
    long_description    = """
    Module for gridding OMI data from Level 2 to Level 3.
    """,
    url                 = '',
    download_url        = '',
    author              = 'Gerrit Kuhlmann',
    author_email        = 'gerrit.kuhlmann@gmail.com',
    platforms           = ['any'],
    license             = 'GNU3',
    keywords            = ['python', 'OMI'],
    classifiers         = ['Development Status :: Beta',
                           'Intended Audiance :: Science/Research',
                           'License :: GNU version 3',
                           'Operating System :: OS Independent'
                          ],
    packages            = ['omi'],
    package_data        = {'omi': files},
    ext_modules         = extensions,
    include_dirs        = [numpy.get_include()]
)
