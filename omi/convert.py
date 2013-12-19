#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


from __future__ import division
import numpy as np
import numpy.ma as ma


def rect2sphere(vector, degree=True):
    """\
    Convert vector (x,y,z) from rect to sphere coordinates. If degree is 
    ``True`` the unit will be in degree.

    Examples
    --------
    >>> convert.rect2sphere([1,1,1], degree=False)
    array([ 0.78539816,  0.61547971,  1.73205081])

    >>> convert.rect2sphere(numpy.array([[1,2],[1,0],[1,3]]), degree=True) 
    array([[ 45.        ,   0.        ],
           [ 35.26438968,  56.30993247],
           [  1.73205081,   3.60555128]])

    """
    x, y, z = vector

    r = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan2(y,x)
    lat = np.arcsin(z/r)

    if degree:
        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)

    return ma.concatenate([ lon[np.newaxis], lat[np.newaxis], r[np.newaxis] ])


def sphere2rect(vector, degree=True):
    """\
    Convert vector (lon, lat, r) from sphere to rect coordinates. If degree
    is True, the unit of vector has to be in degree.

    Examples
    --------
    >>> convert.sphere2rect([120, 30, 1], degree=True)
    array([-0.4330127,  0.75     ,  0.5      ])
    """
    lon, lat, r = vector

    if degree:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)

    return ma.concatenate([
        (r * np.cos(lat) * np.cos(lon))[np.newaxis],
        (r * np.cos(lat) * np.sin(lon))[np.newaxis],
        (r * np.sin(lat))[np.newaxis]
    ])




