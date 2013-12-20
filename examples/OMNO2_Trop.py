# /usr/bin/env python
# coding: utf-8

from datetime import datetime

import numpy as np
import numpy.ma as ma

import omi


#########################################################################
# This is an example script for gridding OMI data using the Python OMI 
# package (please start reading this file at the bottom after the
# "if __name__ == '__main__'" line.
#########################################################################



NAME2DATASET_PIXEL = {}
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields',
    ['TiledArea', 'TiledCornerLatitude', 'TiledCornerLongitude',
     'FoV75Area', 'FoV75CornerLatitude', 'FoV75CornerLongitude'],
    NAME2DATASET_PIXEL
)
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields',
    ['Latitude', 'Longitude', 'SpacecraftAltitude',
     'SpacecraftLatitude', 'SpacecraftLongitude'],
    NAME2DATASET_PIXEL
)

NAME2DATASET_NO2 = {}
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/ColumnAmountNO2/Data Fields',
    ['CloudRadianceFraction', 'CloudPressure', 'ColumnAmountNO2Trop',
     'ColumnAmountNO2TropStd', 'RootMeanSquareErrorOfFit',
     'VcdQualityFlags', 'XTrackQualityFlags'],
    NAME2DATASET_NO2
)
omi.he5.create_name2dataset(
    '/HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields',
    ['SolarZenithAngle', 'Time'],
    NAME2DATASET_NO2
)



def preprocessing(gridding_method, Time, ColumnAmountNO2Trop,
    ColumnAmountNO2TropStd, FoV75Area, CloudRadianceFraction,
    RootMeanSquareErrorOfFit, SolarZenithAngle, VcdQualityFlags,
    XTrackQualityFlags, **kwargs):

    # mask of bad values
    mask = ColumnAmountNO2Trop.mask | ColumnAmountNO2TropStd.mask

    # mask low quality data
    mask |= RootMeanSquareErrorOfFit > 0.0003
    mask |= SolarZenithAngle > 85
    mask |= VcdQualityFlags % 2 != 0
    mask |= XTrackQualityFlags

    # set invalid cloud cover to 100% -> smallest weight
    CloudRadianceFraction[CloudRadianceFraction.mask] = 1.0

    # values and errors
    values = ma.array(ColumnAmountNO2Trop, mask=mask)
    errors = ma.array(ColumnAmountNO2TropStd, mask=mask)

    # weight based on stddev and pixel area (see Wenig et al., 2008)
    stddev = 1.5e15 * (1.0 + 3.0 * ma.array(CloudRadianceFraction, mask=mask))
    area = FoV75Area.reshape(1, FoV75Area.size)
    area = area.repeat(ColumnAmountNO2Trop.shape[0], axis=0)

    if gridding_method.startswith('psm'):
        weights = ma.array(1.0 / area, mask=mask)
    else:
        weights = ma.array(1.0 / (area * stddev**2), mask=mask)

    return values, errors, stddev, weights




def main(start_date, end_date, gridding_method, grid_name, data_path):

    # 1. Define a grid
    # (a) by giving lower-left and upper-right corner
    grid = omi.Grid(
        llcrnrlat=19.6, urcrnrlat=25.6,
        llcrnrlon=108.8, urcrnrlon=117.6, resolution=0.01
    )
    # (b) or by reading this data from a JSON file
    #    (the default file can be found in omi/data/gridds.json)
    grid = omi.Grid.by_name(grid_name)

    # 2. Define parameter for PSM
    #    - gamma (smoothing parameter)
    #    - rho_est (typical maximum value of distribution)
    rho_est = 4e16
    if gridding_method == 'psm':
        # gamma is computed as function of pixel overlap
        gamma = omi.compute_smoothing_parameter(1.0, 10.0)

    # 3. Define a mapping which maps a key to the path in the
    #    HDF file. The function
    #    >>> omi.he5.create_name2dataset(path, list_of_dataset_names, dict)
    #    can be helpful (see above).
    name2datasets = [NAME2DATASET_NO2, NAME2DATASET_PIXEL]

    # 4a) data in OMI files can be read by
    # >>> data = omi.he5.read_datasets(filename, name2dataset)

    # 4b) or by iterating over orbits from start to end date at the following
    #   location: 
    #       os.path.join(data_path, product, 'level2', year, doy, '*.he5')
    #
    #   (see omi.he5 module for details)
    products = ['OMNO2.003', 'OMPIXCOR.003']
    for timestamp, orbit, data in omi.he5.iter_orbits(
            start_date, end_date, products, name2datasets, data_path
        ):

        # 5) Check for missing corner coordinates, i.e. the zoom product,
        #    which is currently not supported
        if (data['TiledCornerLongitude'].mask.any() or
            data['TiledCornerLatitude'].mask.any()
        ):
            continue

        # 6) Clip orbit to grid domain
        lon = data['FoV75CornerLongitude']
        lat = data['FoV75CornerLatitude']
        data = omi.clip_orbit(grid, lon, lat, data, boundary=(2,2))

        if data['ColumnAmountNO2Trop'].size == 0:
            continue

        # 7) Use a self-written function to preprocess the OMI data and
        #    to create the following arrays MxN:
        #    - measurement values 
        #    - measurement errors (currently only CVM grids errors)
        #    - estimate of stddev (used in PSM)
        #    - weight of each measurement
        #    (see the function preprocessing for an example)
        values, errors, stddev, weights = preprocessing(gridding_method, **data)
        missing_values = values.mask.copy()

        if np.all(values.mask):
            continue


        # 8) Grid orbit using PSM or CVM:
        print 'time: %s, orbit: %d' % (timestamp, orbit)
        if gridding_method == 'psm':
            grid = omi.psm_grid(grid,
                data['Longitude'], data['Latitude'],
                data['TiledCornerLongitude'], data['TiledCornerLatitude'],
                values, errors, stddev, weights, missing_values,
                data['SpacecraftLongitude'], data['SpacecraftLatitude'],
                data['SpacecraftAltitude'],
                gamma[data['ColumnIndices']],
                rho_est
            )
        else:
            grid = omi.cvm_grid(grid, data['FoV75CornerLongitude'], data['FoV75CornerLatitude'],
            values, errors, weights, missing_values)


    # 9) The distribution of values and errors has to be normalised
    #    with the weight.
    grid.norm()

    # 10) The Level 3 product can be saved as HDF5 file
    #     or converted to an image (requires matplotlib and basemap)
    grid.save_as_he5('test_%s.he5' % gridding_method)
    grid.save_as_image('test_%s.png' % gridding_method, vmin=0, vmax=rho_est)

    # 11) It is possible to set values, errors and weights to zero.
    grid.zero()



if __name__ == '__main__':

    # "data_path" is the root path to your OMI data. Please change it for
    # you settings. 
    #
    # The OMI files are assumed to be location at:
    #    "{data_path}/{product}/level2/{year}/{doy}/*.he5"
    #
    # For example:
    #    "/home/gerrit/Data/OMI/OMNO2.003/level2/2006/123/*.he5"
    # 
    data_path = '/home/gerrit/Data/OMI'

    # The start and end date of Level 2 data you want to grid. The end_date
    # is NOT included!
    start_date = datetime(2006,10,5)
    end_date = datetime(2006,10,6)

    # Name of the level 3 grid
    grid_name = 'prd'

    # Call main function twice to grid data using CVM and PSM
    main(start_date, end_date, 'cvm', grid_name, data_path)
    main(start_date, end_date, 'psm', grid_name, data_path)





