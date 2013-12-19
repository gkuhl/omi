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

from __future__ import division
import os

import numpy as np

import convert

#import pyximport; pyximport.install()
import cgrate


EARTH_RADIUS = 6378.5 # km


def compute_pixel_size(tiled_lon, tiled_lat, only_dx=False):
    """\
    Compute size of OMI pixels in along- and
    across-track direction.

    If `only_dx` set dy to be constant: 13.5km.
    """
    _, m, n = tiled_lon.shape
    dx = np.ones((m,n))
    dy = 13.5 * np.ones((m,n))

    for i in xrange(n):
        for j in xrange(m):
            dx[j,i] = 0.5 * (
                cgrate.geo_distance(tiled_lon[0,j,i], tiled_lat[0,j,i],
                                    tiled_lon[1,j,i], tiled_lat[1,j,i]) +
                cgrate.geo_distance(tiled_lon[3,j,i], tiled_lat[3,j,i],
                                    tiled_lon[2,j,i], tiled_lat[2,j,i])
            )

    if not only_dx:

        for i in xrange(n):
            for j in xrange(m):
                dy[j,i] = 0.5 * (
                    cgrate.geo_distance(tiled_lon[0,j,i], tiled_lat[0,j,i],
                                        tiled_lon[3,j,i], tiled_lat[3,j,i]) +
                    cgrate.geo_distance(tiled_lon[1,j,i], tiled_lat[1,j,i],
                                        tiled_lon[2,j,i], tiled_lat[2,j,i])
                )

    return dx,dy




def compute_distance(lon, lat, sc_lon, sc_lat, sc_alt):
    """\
    Compute distance between ground pixels (lon, lat) and
    spacecraft (sc_lon, sc_lat, sc_alt) in kilometres.
    """
    sc_alt = 1.0e-3 * sc_alt + EARTH_RADIUS

    n_rows, n_cols = lon.shape

    ground_points = convert.sphere2rect((lon, lat, EARTH_RADIUS))
    sat_positions = convert.sphere2rect((sc_lon, sc_lat, sc_alt))
    sat_positions = sat_positions[:,:,np.newaxis].repeat(n_cols, axis=2)

    d = sat_positions - ground_points
    distance = np.sqrt((d**2).sum(axis=0))

    return distance




