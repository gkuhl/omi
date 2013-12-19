#! /usr/bin/env python
# cython: profile=True

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


import cython
import numpy as np
cimport numpy as np

from libc.math cimport sin, cos, atan2, sqrt
from libc.math cimport ceil, floor


cdef extern from "math.h":
    bint isnan(double x)


@cython.boundscheck(False)
cpdef np.ndarray[np.int_t, ndim=1] compute_line(
        np.ndarray[np.float64_t] x,
        np.ndarray[np.float64_t] lon,
        np.ndarray[np.float64_t] lat,
        int start, int step):
    """\
    compute_line(
        np.ndarray[np.float32_t] x,
        np.ndarray[np.float32_t] lon,
        np.ndarray[np.float32_t] lat,
        int start, int step
    )

    Computes 'y' values for 'x' using the corners defined by 'lon' and 'lat'
    The first corner is given by 'start'. The next corner is given by the
    'step' direction, which should be '+1' for the lower line and '-1' for
    the upper line.
    """

    cdef np.ndarray[np.int_t] y
    cdef double slope, intercept
    cdef int i,k,l

    y = np.zeros(x.size, dtype='int')
    k = start
    l = (start + step) % 4
    if abs(lon[k] - lon[l]) < 1e-9:
        slope = 0.0
    else:
        slope = (lat[l] - lat[k]) / (lon[l] - lon[k])
    intercept = lat[k] - slope * lon[k]

    for i in xrange(x.size):
        if x[i] > lon[l]:
            k = (k + step) % 4
            l = (k + step) % 4

            if abs(lon[k] - lon[l]) < 1e-9:
                slope = 0.0
            else:
                slope = (lat[l] - lat[k]) / (lon[l] - lon[k])
            intercept = lat[k] - slope * lon[k]

        if step == -1: # upper line
            y[i] = int(floor(slope * x[i] + intercept))
        else: # +1; lower line
            y[i] = int(ceil(slope * x[i] + intercept))

    return y



@cython.boundscheck(False)
cpdef np.ndarray[np.int_t, ndim=2] compute_boundaries(
        np.ndarray[np.float64_t] lon,
        np.ndarray[np.float64_t] lat
    ):
    """\
    Compute lower and upper boundary using of pixel and x indices.
    """
    cdef int start, end
    cdef np.ndarray[np.float64_t] x
    cdef np.ndarray[np.int_t, ndim=2] boundaries

    start = np.argmin(lon)
    end = np.argmax(lon)

    x = np.arange(ceil(lon[start]), floor(lon[end] + 1.0), dtype='float64')

    boundaries = np.zeros((3,x.size), dtype=int)
    boundaries[0] = np.array(x, dtype=int)
    boundaries[1] = compute_line(x, lon, lat, start, +1)
    boundaries[2] = compute_line(x, lon, lat, start, -1)

    return boundaries




def draw_pixel_by_psm(
        np.ndarray[np.float64_t, ndim=2] grid_values,
        np.ndarray[np.float64_t, ndim=2] grid_weights,
        np.ndarray[np.float64_t] grid_lon,
        np.ndarray[np.float64_t] grid_lat,
        np.ndarray[np.float64_t] drawing_lon,
        np.ndarray[np.float64_t] drawing_lat,
        np.ndarray[np.float64_t] pixel_lon,
        np.ndarray[np.float64_t] pixel_lat,
        double p00, double p10, double p01, double p11,
        double qx0, double qx1, double qy0, double qy1,
        double d, double rho, double weight,
        double alpha, double beta
    ):
    cdef int u, v, k
    cdef double value
    cdef np.ndarray[np.float64_t] sides
    cdef np.ndarray[np.int_t, ndim=2] boundaries

    boundaries = compute_boundaries(drawing_lon, drawing_lat)

    if boundaries[0].size > 10000:
        return # FIXME: fast fix for pixels on 180E/W

    sides = np.empty(4)
    sides = pixel_side_lengths(pixel_lon, pixel_lat)

    for k in xrange(boundaries[0].size):
        u = boundaries[0,k]

        for v in xrange(boundaries[1,k], boundaries[2,k]+1):

            if u >= 0 and u < grid_lon.size and v >= 0 and v < grid_lat.size:

                coords = compute_coordinates(grid_lon[u], grid_lat[v], pixel_lon, pixel_lat, sides)

                if coords[1] >= alpha or coords[1] <= beta:

                    value = psm_formula(coords[0],coords[1],
                        p00, p10, p01, p11, qx0, qx1, qy0, qy1, d
                    )
                    grid_values[u,v] += (weight * value)
                    grid_weights[u,v] += weight



def draw_orbit(
        np.ndarray[np.float64_t, ndim=1] grid_lon,
        np.ndarray[np.float64_t, ndim=1] grid_lat,
        np.ndarray[np.float64_t, ndim=2] grid_values,
        np.ndarray[np.float64_t, ndim=2] grid_errors,
        np.ndarray[np.float64_t, ndim=2] grid_weights,
        np.ndarray[np.float64_t, ndim=3] lattice_lon,
        np.ndarray[np.float64_t, ndim=3] lattice_lat,
        np.ndarray[np.float64_t, ndim=3] geo_lon,
        np.ndarray[np.float64_t, ndim=3] geo_lat,
        np.ndarray[np.float64_t, ndim=2] rho,
        np.ndarray[np.float64_t, ndim=2] errors,
        np.ndarray[np.float64_t, ndim=2] weights,
        np.ndarray[np.int_t, ndim=2] mask,
        np.ndarray[np.float64_t, ndim=2] alpha,
        np.ndarray[np.float64_t, ndim=2] beta,
        np.ndarray[np.float64_t, ndim=2] p,
        np.ndarray[np.float64_t, ndim=2] d,
        np.ndarray[np.float64_t, ndim=2] qx,
        np.ndarray[np.float64_t, ndim=2] qy,
        method='psm'
    ):
    """\
    Main function for gridding using the Parabolic Spline Method.
    """

    cdef int i,j,u,v
    cdef int n_swaths, n_pixels, n_cols, n_rows

    n_swaths = lattice_lon.shape[1]
    n_pixels = lattice_lon.shape[2]
    n_cols = grid_values.shape[0]
    n_rows = grid_values.shape[1]

    if method == 'psm':
        for i in xrange(n_pixels):
            for j in xrange(n_swaths):
                draw_pixel_by_psm(
                    grid_values, grid_weights, grid_lon, grid_lat,
                    lattice_lon[:,j,i], lattice_lat[:,j,i],
                    geo_lon[:,j,i], geo_lat[:,j,i],
                    p[j,i], p[j,i+1], p[j+1,i], p[j+1,i+1],
                    qx[j,i], qx[j+1,i], qy[j,i], qy[j,i+1], d[j,i],
                    rho[j,i], weights[j,i],
                    alpha[j,i], beta[j,i]
                )

    elif method == 'cvm':
        for i in xrange(n_pixels):
            for j in xrange(n_swaths):
                if mask[j,i] == False:
                    draw_pixel_by_cvm(
                        grid_values, grid_errors, grid_weights, grid_lon, grid_lat,
                        lattice_lon[:,j,i], lattice_lat[:,j,i],
                        rho[j,i], errors[j,i], weights[j,i]
                    )


    return grid_values




cpdef pixel_side_lengths(np.ndarray[np.float64_t] lon, np.ndarray[np.float64_t] lat):
    """\
    Compute the four side lenght of a pixel in km.
    """
    cdef np.ndarray[np.float64_t] sides

    sides = np.empty(4)

    # compute: ab, bc, cd, da
    sides[0] = geo_distance(lon[0], lat[0], lon[1], lat[1])
    sides[1] = geo_distance(lon[1], lat[1], lon[2], lat[2])
    sides[2] = geo_distance(lon[2], lat[2], lon[3], lat[3])
    sides[3] = geo_distance(lon[3], lat[3], lon[0], lat[0])

    return sides



cpdef triangle_altitude(double a, double b, double c):
    """\
    Compute altitude at point C of triangle with side length a,b,c.
    """
    cdef double a2, b2, c2

    if a+b <= c:
        return 0.0

    a2 = a**2
    b2 = b**2
    c2 = c**2

    return sqrt(2.0 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2 * a2 + b2 * b2 + c2 * c2)) / (2.0 * c)


cpdef compute_coordinates(double clon, double clat,
    np.ndarray[np.float64_t] lon, np.ndarray[np.float64_t] lat,
    np.ndarray[np.float64_t] sides
    ):
    """\
    Compute s and t coordinates:
        clon, clat:  current point
        lon, lat:    corners of pixel
        sides:       length of the four sides of a pixel

    Returns:
        dimensionless coords s and t

    """
    cdef double a,b,c, s, t
    cdef double s_left, s_right, t_up, t_low
    cdef np.ndarray[np.float64_t] coords

    coords = np.empty(2)

    a = geo_distance(clon, clat, lon[0], lat[0])
    b = geo_distance(clon, clat, lon[1], lat[1])
    c = geo_distance(clon, clat, lon[2], lat[2])
    d = geo_distance(clon, clat, lon[3], lat[3])

    s_left = triangle_altitude(a, d, sides[3])
    s_right = triangle_altitude(c, b, sides[1])

    t_up = triangle_altitude(d, c, sides[2])
    t_low = triangle_altitude(a, b, sides[0])

    coords[0] = s_left / (s_left + s_right)
    coords[1] = t_low / (t_low + t_up)

    return coords



cpdef geo_distance(double lon0, double lat0, double lon1, double lat1):
    """
    geo_distance(float lon0, float lat0, float lon1, float lat1)

    Compute distance between to geographical coordinates
    see: http://en.wikipedia.org/wiki/Great-circle_distance
    """
    cdef double d,s,f
    cdef double cd,sd,cf,sf,cs,ss
    cdef double RADIUS, DEG2RAD

    RADIUS = 6371.0
    DEG2RAD = 0.017453292519943295

    d = DEG2RAD * (lon1 - lon0)
    s = DEG2RAD * lat0
    f = DEG2RAD * lat1

    cd = cos(d)
    sd = sin(d)
    cf = cos(f)
    sf = sin(f)
    cs = cos(s)
    ss = sin(s)

    return RADIUS * atan2(
        sqrt((cf * sd)**2 + (cs * sf - ss * cf * cd)**2),
        ss * sf + cs * cf * cd
    )


cpdef psm_formula(double s, double t, double p00, double p10, double p01, double p11,
    double dx0, double dx1, double dy0, double dy1, d):
    """\
    Compute quadratic surface spline f(s,t) with
    p00, p01, p10, p11, dx0, dx1, dy0, dy1 and d.
    """
    cdef double f

    f =  (1-s) * (1-t) * (1 - 3*s - 3*t + 9*s*t) * p00
    f += s*(1-t) * (-2 + 3*s + 6*t - 9*s*t) * p10
    f += (1-s) * t * (-2 + 6*s + 3*t - 9*s*t) * p01
    f += s*t * (4 - 6*s - 6*t + 9*s*t) * p11

    f += 6 * (1-s) * (1-t) * (s*(1-3*t) * dx0 + t * (1 - 3*s) * dy0)
    f += 6*s*t * ( (1-s)* (3*t-2) * dx1 + (1-t)*(3*s-2)* dy1  )

    f += 36*s * (1-s) * t * (1-t) * d

    return f


def draw_pixel_by_cvm(
        np.ndarray[np.float64_t, ndim=2] grid_values,
        np.ndarray[np.float64_t, ndim=2] grid_errors,
        np.ndarray[np.float64_t, ndim=2] grid_weights,
        np.ndarray[np.float64_t] grid_lon,
        np.ndarray[np.float64_t] grid_lat,
        np.ndarray[np.float64_t] drawing_lon,
        np.ndarray[np.float64_t] drawing_lat,
        double rho, double error, double weight,
    ):
    cdef int u, v, k
    cdef double value
    cdef np.ndarray[np.int_t, ndim=2] boundaries

    boundaries = compute_boundaries(drawing_lon, drawing_lat)

    if boundaries[0].size > 10000:
        return # FIXME: fast fix for pixels on 180E/W

    for k in xrange(boundaries[0].size):
        u = boundaries[0,k]

        for v in xrange(boundaries[1,k], boundaries[2,k]+1):

            if u >= 0 and u < grid_lon.size and v >= 0 and v < grid_lat.size:
                grid_values[u,v] += (weight * rho)
                grid_errors[u,v] += (weight**2 * error**2)
                grid_weights[u,v] += weight



