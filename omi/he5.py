#! /usr/bin/env python
# coding: utf-8

# This file is part of the OMI Python package.
#
# The OMI Python package is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# The OMI Python Package is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the OMI Python Package. If not, see
# <http://www.gnu.org/licenses/>.

from datetime import datetime, timedelta

import glob
import errno
import itertools
import os

import h5py
import numpy as np
import numpy.ma as ma
import scipy.sparse as sparse


def iter_dates(start, end, step=1):
    """\
    Iterator over dates from start to end date. The end point
    is not included!

    Parameter
    ---------
    start, end : datetime.date or datetime.datetime
        Interval of days from "start" to "end".

    step : integer
        step size, default = 1

    Example
    -------
    >>> from datetime import date
    >>> for d in iter_dates(date(2010,1,1), date(2010,1,3)):
    ...     print d
    ...
    2010-01-01
    2010-01-02
    """
    current = start
    while current < end:
        yield current
        current += timedelta(days=step)



def parse_filename(filename):
    """\
    Parse OMI data filename to extract and return:
    - satellite name
    - product name
    - measurement time
    - algorithm version
    - processing time
    """
    filename, extension = os.path.splitext(os.path.basename(filename))
    satellite, product, time, version = filename.split('_')

    # orbit and measurement time
    meas_time, orbit = time.split('-o')
    orbit = int(orbit)
    meas_time = datetime.strptime(meas_time, '%Ym%m%dt%H%M')

    # version and processing time
    version, proc_time = version.split('-')
    try:
        proc_time = datetime.strptime(proc_time, '%Ym%m%dt%H%M%S')
    except ValueError:
        print('Warning: Invalid process time (%s)' % proc_time)
        proc_time = None

    return satellite, product, meas_time, orbit, version, proc_time




def iter_filenames(start_date, end_date, products, data_path):
    """\
    Iterator over OMI level 2 filenames from start to end date for a given
    product. The end date is NOT included!

    The OMI files are assumed to be location at:
        "{data_path}/{product}/level2/{year}/{doy}/*.he5"

    For example:
        "/home/gerrit/Data/OMI/OMNO2.003/level2/2006/123/*.he5"

    Parameter
    ---------
    start_date, end_date : datetime.datetime
        The files are returned from start to end date.
        (end_date is NOT included)

    products : list
        list of products, e.g. ['OMNO2.003', 'OMPIXCOR.003']

    data_path : string
        path to data folder


    """
    for date in iter_dates(start_date, end_date):
        year, doy = date.strftime('%Y'), date.strftime('%j')

        filenames = [
            sorted(glob.glob(
                os.path.join(data_path, product, 'level2', year, doy, '*.he5')
            ))
            for product in products
        ]

        for filename in zip(*filenames):
            yield filename


def write_datasets(filename, data, mode='w'):
    """\
    Writes data as HDF 5 file. Creates path if it does
    not exists. Can handle sparse matrices.

    Parameter
    ---------
    filename : string
        filename of HDF5 file

    data : list of tuples
        datasets to be written to HDF5 file as:
        "[(name, array), (name, array), ...]

    mode : string
        file mode (default overwrite 'w')
        use 'a' to append to existing file

    """
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise error

    with h5py.File(filename, mode) as fobj:
        for name, value in data:
            if sparse.issparse(value):
                d = fobj.create_dataset(name, shape=value.shape, fillvalue=0.0, compression='gzip')

                for row, col in itertools.izip( *value.nonzero() ):
                    d[row, col] = value[row, col]

            else:
                fobj.create_dataset(name, data=value, compression='gzip')



def read_datasets(filename, name2dataset):
    """\
    Reads OMI datasets from file as dictionary with
    structure: name -> np.ma.ndarray.

    HDF attributes are used to scale ('ScaleFactor')
    shift ('Offset') values and mask missing values
    ('_FillValue' and 'MissingValue').

    Parameter
    ---------
    filename : string
        HDF5 filename

    name2dataset : dict
        dictionary which gives `name` -> path of dataset

    Returns
    -------
    dictionary with structure: name -> np.ma.ndarray.

    """
    data = dict()

    with h5py.File(filename, 'r') as f:
        for name, path in name2dataset.iteritems():
            field = f.get(path, None)

            if field is None:
                print 'Warning: No %s in %s.' % (path, filename)
                data[name] = None
            else:
                fill_value = field.attrs.get('_FillValue', None)
                missing_value = field.attrs.get('MissingValue', None)
                scale = field.attrs.get('ScaleFactor', 1.0)
                offset = field.attrs.get('Offset', 0.0)

                if fill_value is None and missing_value is None:
                    raise ValueError('Missing `_FillValue` or `MissingValue` in %s in %s' % (path, filename))

                value = field.value
                mask = (field.value == fill_value) | (field.value == missing_value)

                if fill_value < -1e25 or missing_value < -1e25:
                    mask |= (field.value < -1e25)

                if abs(scale - 1.0) > 1e-12:
                    value = value * scale

                if abs(offset - 0.0) > 1e-12:
                    value = value + offset

                if value.dtype == np.float32:
                    data[name] = ma.array(value, mask=mask, dtype=np.float64)
                else:
                    data[name] = ma.array(value, mask=mask, dtype=value.dtype)

    return data



def create_name2dataset(path, names, mapping=None):
    """\
    Function to create `name2dataset` dictionary as used by
    `he5.read_datasets`.

    Parameter
    ---------
    path : string
        path to datasets

    names : list of strings
        name of each dataset

    mapping : dictionary (default None)
        A `name2dataset` to which `name -> os.path.join(path, name)`
        is added. If None an empty dictionary will be created.

    Returns
    -------
    The dictionary mapping.

    """
    if mapping is None:
        mapping = {}

    mapping.update(
        dict((name, '/'.join([path,name])) for  name in names)
    )

    return mapping



def iter_orbits(start_date, end_date, products, name2datasets, data_path):
    """\
    Iterator over OMI data for each orbit (level 2).

    Parameter
    ---------
    start_date, end_date : datetime
        start and end date for which data are returned. The end date
        is NOT included!

    products : list of strings
        list OMI product names, e.g. ['OMNO2.003', 'OMPIXCOR.003']

    name2datasets : list of dictionaries
        mapping used to read data from file, see `he5.read_datasets`
        for details.

    data_path : string
        path to OMI data, see `he5.iter_filenames` to see expected
        file structure for OMI products.

    Returns
    -------
    timestamp : datetime
        start time of orbit

    orbit : integer
        orbit number

    data : dictionary
        dictionary of structure: name->datasets
        See `he5.read_datasets` for details.


    """
    for filenames in iter_filenames(start_date, end_date, products, data_path):

        data = {}
        for filename, name2dataset in zip(filenames, name2datasets):
            _, _, timestamp, orbit, _, _ = parse_filename(filename)
            data.update(read_datasets(filename, name2dataset))

        yield timestamp, orbit, data






