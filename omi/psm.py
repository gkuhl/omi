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
import scipy as sp
import scipy.integrate
import scipy.sparse as sparse
import scipy.sparse.linalg as s_linalg


# constants and satllite parameter (Aura specific)
EARTH_RADIUS = 6378.5
ORBIT_PERIOD = 100.0 * 60.0
GROUND_SPEED = 2.0 * np.pi * EARTH_RADIUS / ORBIT_PERIOD


PACKAGE_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')


class MMatrixLUT(object):
    def __init__(self, path):
        """\
        Look up table for rows of meaurement matrix M.
        """
        self.path = path
        self._lut = {}
        self.load_lut()


    def load_lut(self):
        """\
        Load LUT for different indices j=0,1,...,m-2,m-1.
        """

        for index in [0, 1, None, -2, -1]:
            if index is None:
                filename = 'measurement_equation.dat'
            else:
                filename = 'measurement_equation_{0:+d}.dat'.format(index)

            filename = os.path.join(self.path, filename)
            data = np.loadtxt(filename)

            self._lut[index] = dict((int(d[0]), d[1:]) for d in data)


        # load y-direction limits
        filename = os.path.join(self.path, 'y_limits.dat')
        data = np.loadtxt(filename)
        self._limits = dict((int(d[0]), d[1]) for d in data)


    def get_limit(self, distance):
        distance = int(round(distance))
        return self._limits[distance]


    def get_values(self, distance, j, m):

        distance = int(round(distance))

        if j == 0:
            values = self._lut[0][distance]
        elif j == 1:
            values = self._lut[1][distance]
        elif j == m-2:
            values = self._lut[-2][distance]
        elif j == m-1:
            values = self._lut[-1][distance]
        else:
            values = self._lut[None][distance]

        return values[np.isfinite(values)]





def norm_box_function(x, half_width):
    """\
    A normilised box function on `x` with given `half_width`.
    """
    mask = (-half_width <= x) & (x <= half_width)
    box = np.zeros_like(x)
    box[mask] = 1.0
    box /= box.sum()

    return box


def instrument_function(y, distance, exposure_time):
    """\
    Compute instrument function W(y) for given `distance`
    and `exposure_time`.

    Parameters
    ''''''''''
    y : array_like (or None)
        coordinates with uniform spacing dy; if y is None
        appropiate range will be estimated

    distance : float
        distance between instrument and ground

    exposure_time : float
        instrument exposure time (e.g. OMI 2.0 seconds)

    Returns
    '''''''
    y : np.ndarray
        along-track coordinates

    W : np.ndarray
        instrument function W(y)
    """

    # instantaneous FWHM and scaling coefficient of instrument function
    iFWHM = 2.0 * distance * np.tan( np.deg2rad(0.5) )
    c = -np.log(0.5) / (0.5 * iFWHM)**4

    # estimate suitable range for y, if necessary
    if y is None:
        iFWHM = 2.0 * distance * np.tan( np.deg2rad(0.5) )
        y_width = iFWHM + GROUND_SPEED * exposure_time
        y = np.linspace(-y_width, y_width, 5001)

    # convolute slit function with box function
    dy = y[1] - y[0]
    box = norm_box_function(y, 0.5 * GROUND_SPEED * exposure_time)
    W = np.convolve( np.exp(-c*y**4), box, 'same')
    W /= (W * dy).sum()

    return y, W



def create_knot_set(center, dy, y_min, y_max):
    """\
    Compute location of knots with respect to the center
    of the current pixel:
        [..., -0.5 * dy[center], +0.5 * dy[center], ...]
    and limited by `y_min` and `y_max`.
    """

    yp = [ -0.5 * dy[center], 0.5 * dy[center] ]
    indices = [center, center+1]

    for i in xrange(center+1, dy.size):
        if yp[-1] >= y_max:
            break
        yp.append(yp[-1] + dy[i])
        indices.append(indices[-1] + 1)

    for i in xrange(center-1, -1, -1):
        if yp[0] <= y_min:
            break
        yp.insert(0, yp[0] - dy[i])
        indices.insert(0, indices[0] - 1)

    return np.asarray(yp), indices


def compute_fov_limits(y, W, area_under_curve):
    """\
    Compute lower and upper boundary of instrument function based on
    'area under curve'.
    """
    y_min = y[W.cumsum() >= 0.5 * (1.0 - area_under_curve) ][0]
    y_max = -y_min

    return y_min, -y_min




def look_up_M_matrix_row(row, dy, distance, lut):
    """\
    Get row entries in measurement matrix M from LUT and compute
    coverage for neighbouring tiled pixels by instrument
    function.

    Parameter
    ---------
    row : integer
        index of current row

    dy : array_like, shape(M,)
        pixel size

    distance : float or int
        distance between spacecraft and ground pixel

    lut : psm.MMAatrixLUT
        look-up table for row entries

    Returns
    -------
    values : array_like
        entries for M matrix in row

    columns : array_like
        column indices for `values` in row

    coverage : dictionary
        coverage of neighbouring pixels by
        instrument function
    """

    y_max = lut.get_limit(distance)
    y_min = -y_max

    yp, indices = create_knot_set(row, dy, y_min, y_max)

    values = lut.get_values(distance, row, dy.size)
    columns = np.arange(2*indices[0], 2*indices[0] + values.size)

    coverage = compute_overlap(yp, y_min, y_max, indices)

    return values, columns, coverage



def compute_M_matrix_row(row, dy, distance, exposure_time, area_under_curve):
    """\
    Compute row entries of measurement matrix M by
    numerical integrations.

    Parameter
    ---------
    row : integer
        index of current row

    dy : array_like, shape(M,)
        pixel size

    distance : number
        distance between spacecraft and ground pixel

    exposure_time : number
        instrument exposure time (e.g. OMI 2.0 seconds)

    area_under_curve : number
        fraction of area under curve to be used,
        e.g. 0.75 or 0.99

    Returns
    -------
    values : array_like
        entries for M matrix in row

    columns : array_like
        column indices for `values` in row

    coverage : dictionary
        coverage of neighbouring pixels by
        instrument function
    """

    y, W = instrument_function(None, distance, exposure_time)

    y_min, y_max = compute_fov_limits(y, W, area_under_curve)
    yp, indices = create_knot_set(row, dy, y_min, y_max)

    values = np.zeros(2*yp.size-1)
    columns = np.arange(2*indices[0], 2*indices[0] + values.size)

    # compute matrix entries
    for i, (y0, y1) in enumerate(zip(yp[:-1], yp[1:])):

        mask = (y0 < y) & (y < y1)
        t = (y[mask] - y0) / (y1 - y0)

        # base functions
        phi_p0 = (1 - 4 * t + 3 * t**2)
        phi_d = ( 6 * t - 6 * t**2)
        phi_p1 = (-2 * t + 3 * t**2)

        # integration
        for j, phi in enumerate([phi_p0, phi_d, phi_p1]):
            values[2*i+j] += scipy.integrate.simps(W[mask] * phi, y[mask])

    values / values.sum()

    coverage = compute_overlap(yp, y_min, y_max, indices)

    return values, columns, coverage



def compute_overlap(yp, y_min, y_max, indices):
    """\
    Compute coverage of each tiled pixels if instrument
    function lies between `y_min` and `y_max`.

    Parameter
    ---------
    yp : array_like, shape(M+1,)
        location of knots on lattice

    y_min, y_max : number
        minimum and maximum location of instrument
        function in along-track direction

    indices : list of integer
        along-track position of tiled pixels

    Returns
    -------
    A dictionary which gives the coverage from
    right (alpha, full coverage 0) and left (beta,
    full coverage 1).

    """

    # compute boundaries on each interval (full coverage: [0,1])
    alpha = np.zeros(yp.size - 1)
    beta = np.ones(yp.size - 1)
    alpha[0] = (y_min - yp[0])  / (yp[1] - yp[0])
    beta[0] = 0.0

    alpha[-1] = 1.0
    beta[-1] = (y_max - yp[-2]) / (yp[-1] - yp[-2])

    # coverage of W(y) over knot set
    coverage = {}
    for i in xrange(yp.size-1):
        coverage[indices[i]] = [alpha[i], beta[i]]

    return coverage


def update_coverage(total, new):
    """\
    Update dictionary which describes the coverage of each
    tiled pixel by the instrument functions of valid
    measurements.
    """
    for key in new:
        if key in total:
            if new[key][0] < total[key][0]: # alpha: optimal 0
                total[key][0] = new[key][0]
            if new[key][1] > total[key][1]: # beta: optimal 1
                total[key][1] = new[key][1]

        else:
            total[key] = new[key]

    return total


def M_matrix(values, stddev, dy, distances, exposure_time, missing_values,
    area_under_curve=None, lut=None):
    """\
    Compute measurement matrix M (normalised with standard deviation) and
    the coverage by non-missing values over each interval.

    Parameter
    ---------
    values : array_like, shape (M,)
        measurement values

    stddev : array_like, shape (M,)
        standard deviation of measurments

    dy : array_like, shape(M,) or shape(,)
        length of intervals in along-track direction [y_j, y_{j+1}]

    distances : array_like, shape(M,) or shape(,)
        distances between instrument and ground pixel

    exposure_time : number
        instrument exposure time (e.g. OMI 2.0 seconds)

    missing_values : array_like, bool, shape(M,N) or None
        Missing measurement values. Missing meausrements will not
        be included in the coverage dictionary (see returns).
        If `None` no missing values.

    area_under_curve : float or None
        fraction of area under instrument function (e.g. 0.75 or 0.99)

    lut : omi.psm.MMatrixLUT object
        Look-up table for entries of matrix M (kappa)

    Returns
    -------
    M : sparse.csr_matrix
        measurement matrix normalised with standard deviation

    coverage : dictionary
        coverage of tiled pixels by instrument functions
        of non-missing measurements

    """

    coverage = {}

    if dy.size == 1:
        dy = dy.repeat(values.size)

    if distances.size == 1:
        distances = distances.repeat(values.size)

    # create CSR matrix
    data = []
    indices = []
    indptr = [0]

    for row in xrange(values.size):
        if np.isfinite(values[row]):

            if lut is None:
                # numerical integration (simpson)
                row_entries, columns, m_coverage = compute_M_matrix_row(row, dy,
                        distances[row], exposure_time, area_under_curve
                )

            else:
                row_entries, columns, m_coverage = look_up_M_matrix_row(row, dy,
                        distances[row], lut
                )

            # add  only valid (i.e. measured) values to coverage
            if not missing_values[row]:
                coverage = update_coverage(coverage, m_coverage)

            data.append(row_entries / stddev[row])
            indices.append(columns)
            indptr.append(indptr[-1] + row_entries.size)

    M = sparse.csr_matrix((
        np.concatenate(data),
        np.concatenate(indices),
        np.asarray(indptr)
    ), shape=(len(indptr)-1, 2*values.size+1))


    return M, coverage



def C1_matrix(h):
    """\
    Create C1 continity matrix for x = [p0, d0, p1, ...] with
    natural boundary condition. The pixel size is `h`.
    """
    a = 1.0 / np.asarray(h)

    shape = (a.size+1 , 2*a.size+1)
    C = sparse.lil_matrix(shape)

    C[0,0:3] = [-4.0, +6.0, -2.0]

    for row in xrange(1, a.size):
        col = 2 * row

        C[row, col-2] =         a[row-1]
        C[row, col-1] = -3.0 *  a[row-1]
        C[row, col  ] =  2.0 * (a[row-1] + a[row])
        C[row, col+1] = -3.0 *  a[row]
        C[row, col+2] =         a[row]

    C[a.size, -3:] = [2.0, -6.0, 4.0]

    b = np.zeros(a.size+1)


    return C.tocsr(), b



def L2_matrix(m):
    """\
    Create second-order difference matrix L2
    (defined on mean values d_j only)
    """
    L2 = sparse.lil_matrix((m-2, 2*m+1))

    for row in xrange(m-2):

        col = 2 * (row + 1) + 1

        L2[row, col-2] =  1.0
        L2[row, col]   = -2.0
        L2[row, col+2] =  1.0

    return L2 / 3.0


def B_inv_matrix(stddev, rho_est):
    """\
    Create inverse of diagonal matrix B.
    """
    B_inv = 1.0 / (rho_est * stddev[1:-1].copy())
    B_inv = sparse.dia_matrix((B_inv, 0), shape=(B_inv.size, B_inv.size))

    return B_inv



def penalty_term(stddev, gamma, rho_est):
    """\
    Compute penalty term: gamma * L2.T * B * L2
    """
    B_inv = B_inv_matrix(stddev, rho_est)
    L2 = L2_matrix(stddev.size)

    return gamma * L2.T * B_inv *  L2



def solve_qpp(H, C, g, b=None):
    """\
    Solve quadratic programming problem.

    Parameter
    ---------
    H : sparse matrix
        Hessian matrix

    C : sparse matrix
        C1 continuity matrix

    g : vector
        g = 2y.T S.I M

    Returns
    -------
    parameter vector x of size 2m+1

    """
    m,n = C.shape

    K = sparse.vstack([
        sparse.hstack([H, C.T]),
        sparse.hstack([C, sparse.csr_matrix((m,m))])
    ]).tocsc()

    if b is None:
        b = np.zeros(m)

    b = np.concatenate([-g, b])
    res = s_linalg.spsolve(K, b)

    return res[:n]


def across_track(d, h):
    """\
    Compute parameter vector `p` of tiled parabolic C1
    histospline:

            p = C.I * b

    Parameters
    ''''''''''
    d : array_like, shape(N,)
        mean value over each interval

    h : array_like, shape(N,)
        size of each interval [x_i, x_i+1]

    Returns
    '''''''
    p : np.ndarray, shape(N,)
        parameter vector of coefficients `p`, i.e. values of
        spline at each knot `x_i`

    """
    a = 1.0 / np.asarray(h)

    C = sparse.lil_matrix((a.size+1, a.size+1))
    b = np.zeros(a.size+1)

    # natural boundary condition (left)
    C[0,0:2] = [-4.0, -2.0]
    b[0] = -6.0 * d[0]

    # C1 continuity
    for i in xrange(1, a.size):
        C[i, i-1] = a[i-1]
        C[i, i  ] = 2.0 * (a[i-1] + a[i])
        C[i, i+1] = a[i]

        b[i] = 3.0 * (d[i-1] * a[i-1] + d[i] * a[i])

    # natural boundary condition (right)
    C[a.size, -2:] = [2.0, 4.0]
    b[a.size] = +6.0 * d[a.size - 1]

    # solve using sparse LU decomposition
    p = s_linalg.dsolve.factorized( C.tocsc() )(b)

    return p




def along_track(values, stddev, dy, gamma, distances, exposure_time=2.0,
    auc=0.99, missing_values=None, lut=None, rho_est=None):
    """\
    Compute vector of coefficients `x` of smoothing spline.

    Parameter
    ---------
    values : array_like, shape (N,)
        measurement values.

    stddev : array_like, shape (N,) or shape(,)
        errors of measurments.

    dy : array_like, shape(N,) or shape(,)
        length of intervals [y_i, y_{i+1}]

    gamma : float
        weighting factor

    distances : array_like, shape(N,) or shape(,)
        distances between instrument and ground pixel

    exposure_time : float
        instrument's exposure time (default 2.0 for OMI)

    auc : float
        area under curve (default 0.99)

    valid_rho : array_like, bool, shape(N,) or None
        Valid measurements are True. Non valid meausrements will not be
        included in the coverage dictionary (see returns). Default all
        are True, i.e. all values are used.


    Returns
    ------
    x : ndarray, shape (2M+1,)
        optimal spline parameters [p0, d0, p1, ..., p_N+1]

    alpha, beta : ndarrays, shape(N,)
        As weighting functions are overlapping in neighbouring knot intervals,
        the coefficients α and β are used to describe the coverage
        of each knot interval using the unit interval.

        β describes, how far the interval is covered from 0 (lower boundary)
        into the interval where β=0 is no coverage and β=1 is full coverage.
        A β>1 may occur at the boundary of the knot set, if the weighting
        function is reaching over the largest knot.

        α describes how far the interval is covered from 1 (upper boundary)
        into the interval where α=1 is no coverage and α=0 is full coverage.
        Similar α<0 may occur on the lower boundary of the knot set.

        An interval is fully covered if β>=α (see the following example):

                              |1-α|
                           ,---------,
                    |______.______.__|
                    0      α      β  1
                    `-------------'
                          |β|

        and misses some areas if α>β (indicated by `xxxx`):

                           xxxx
                    |_____.____._____|
                    0     β    α     1
                    `-----'    `-----'
                      |β|       |1-α|

        α and β can be used to draw parts of the surface spline for missing
        or invalid pixels, which are partially covered neighbouring pixels.

    """

    values = np.asarray(values)
    stddev = np.asarray(stddev)
    dy = np.asarray(dy)
    distances = np.asarray(distances)

    if rho_est is None:
        rho_est = 1.0

    if missing_values is None:
        missing_values = np.zeros(values.shape, bool)

    if stddev.size == 1:
        stddev = stddev.repeat(values.size)

    #mask = np.isfinite(values)

    # measurment matrix
    M, coverage = M_matrix(values, stddev, dy, distances, exposure_time,
        missing_values, area_under_curve=auc, lut=lut
    )
    d = np.matrix( values / stddev ).T

    # penalty term
    P = penalty_term(stddev, gamma, rho_est)

    # objective function
    H = 2.0 * (M.T * M + P)
    g = -2.0 * np.array(d.T * M).flatten()
    k = float(d.T * d)

    # C1 constraint
    C, b = C1_matrix(dy)

    # inversion of QPP
    x = solve_qpp(H, C, g, b)

    # compute coverage coefficents alpha and beta
    alpha = np.ones_like(values)
    beta = np.zeros_like(values)
    for i in coverage:
        alpha[i] = coverage[i][0]
        beta[i] = coverage[i][1]

    return x, alpha, beta




def parabolic_spline_algorithm(values, stddev, dx, dy, gamma, rho_est,
    distances, exposure_time=2.0, missing_values=None, area_under_curve=0.99,
    lut=None):
    """\
    Compute coefficients of parabolic surface spline for a function
    rho(x,y) of an OMI-like orbit. The surface spline is defined on
    a lattice with M rows in along-track direction and N columns in
    across-track direction.

    Parameter
    ---------

    values : array_like, shape (M,N)
        measurement values

    stddev : array_like, shape (M,N) or shape(,)
        standard deviation of measurments

    dx : array_like, shape(M,N) or shape(,)
        length of intervals in across-track direction [x_i, x_{i+1}]

    dy : array_like, shape(N,) or shape(,)
        length of intervals in along-track direction [y_j, y_{j+1}]

    gamma : array_like, shape (N,) or shape (,)
        smoothing parameter

    rho_est : float
        estimate of maximum value (for scaling)

    distances : array_like, shape(M,N) or shape(,)
        distances between instrument and ground pixel

    exposure_time : float
        instrument's exposure time (default 2.0 for OMI)

    missing_values : array_like, bool, shape(M,N) or None
        Missing measurement values. Missing meausrements will not
        be included in the coverage dictionary (see returns).
        If `None` no missing values.

    area_under_curve : float
        area under curve (default 0.99)

    lut : omi.psm.MMatrixLUT object
        Look-up table for entries of matrix M (kappa)

    Returns
    -------

    p : array_like, shape(M+1,N+1)
        values of spline on lattice knots

    d : array_like, shape(M,N)
        mean values of each lattice cell

    qx : array_like, shape(M+1,N)
        line integrals of spline on [x_i, x_{i+1}]

    qy : array_like, shape(M,N+1)
        line integrals of spline on [y_j, y_{j+1}]

    alpha : array_like, shape(M,N)
        defines coverage of lattice cells in along-track direction
        (see omi.psm.along_track for details)

    beta : array_like, shape(M,N)
        defines coverage of lattice cells in along-track direction
        (see omi.psm.along_track for details)

    """

    # scale values to avoid large condition numbers
    values /= rho_est
    stddev /= rho_est

    m,n = values.shape

    d = np.empty((m,n))
    qx = np.empty((m+1,n))
    qy = np.empty((m,n+1))
    p = np.empty((m+1,n+1))

    alpha = np.ones_like(values)
    beta = np.zeros_like(values)

    gamma = np.asarray(gamma)
    if gamma.size == 1:
        gamma = gamma.repeat(n)

    distances = np.asarray(distances)
    if distances.size == 1:
        distances = distances * np.ones_like(values)

    stddev = np.asarray(stddev)
    if stddev.size == 1:
        stddev = stddev * np.ones_like(values)

    if missing_values is None:
        missing_values = np.zeros(values.shape, bool)


    # compute d and qx
    for i in xrange(n):
        x, alpha[:,i], beta[:,i] = along_track(values[:,i], stddev[:,i], dy[:,i], gamma[i],
            distances[:,i], exposure_time=exposure_time, missing_values=missing_values[:,i],
            auc=area_under_curve, lut=lut, rho_est=1.0
        )
        qx[:,i] = x[::2].copy()
        d[:,i] = x[1::2].copy()


    # compute qy
    for j in xrange(m):
        qy[j,:] = across_track(d[j,:], dx[j,:])


    # compute p_ij
    p[0,:] = across_track(qx[0,:], dx[0,:])
    p[-1,:] = across_track(qx[-1,:], dx[-1,:])

    for j in xrange(1,m):
        p[j,:] = across_track(qx[j,:], 0.5 * (dx[j-1,:] + dx[j,:])  )


    # scale coefficients up
    p *= rho_est
    d *= rho_est
    qx *= rho_est
    qy *= rho_est

    return p, d, qx, qy, alpha, beta



if __name__ == '__main__':
    pass




