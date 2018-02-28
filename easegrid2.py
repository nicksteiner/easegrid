#! /usr/bin/python
'''
Author 
~~~~~~
n.c.steiner, 2013, 2018

Requires: py-yaml, numpy, pyproj, affine, xarray

'''
import os
import sys
from collections import namedtuple

import affine
import yaml
import numpy as np
import xarray as xr
from pyproj import Proj


class _Grid(object):
    '''EASEGrid grid object template.

    Keywords:
    grid_name -- Subgrid name (e.g. g09, g03)

    ''' 

    _src_path = os.path.dirname(os.path.realpath(__file__))
    _dat_path = os.path.join(_src_path, 'dat')
    _constant = yaml.load(open(os.path.join(_src_path, 'grids.yaml'), 'r').read())
    coord_scale = 1e-05

    def __init__(self, grid_name):
        # get grid name
        try:
            assert grid_name in self.grid_names
        except:
            raise Exception('Valid grid name in {0}'.format(self.grid_names))
        self.grid_name = grid_name
        # set the projection
        self.p = self.p_args
        # set the grid
        _constant = namedtuple('grid_constant', 'size cols rows r0 s0')
        self.constant = _constant(**self._constant[self.name][grid_name])
        # id string unique to grid/proj combinatio
        self.grid_id = self.name + '_' + getattr(self, 'hemi', 'G') + '_' + self.grid_name
        
    '''
    Properties
    ~~~~~~~~~~
    '''

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, val):
        ''' Proj4 object.'''
        val = dict(zip(val, [getattr(self, v, None) for v in val]))
        self._p = Proj(val)

    @property
    def geotransform(self):
        ''' GDAL formatted geotransform.
        :rtype: list
        '''
        gt0 = self.constant.s0 - self.constant.size * self.constant.cols / 2
        gt3 = self.constant.r0 + self.constant.size * self.constant.rows / 2
        return [gt0, self.constant.size, 0, gt3, 0, -self.constant.size]

    @property
    def transform(self):
        return affine.Affine.from_gdal(*self.geotransform)

    @property
    def itransform(self):
        return ~self.transform

    @property
    def latitude(self):
        if not hasattr(self, '_latitude'):
            self.set_coords()
        return self._latitude
    
    @property
    def longitude(self):
        if not hasattr(self, '_longitude'):
            self.set_coords()
        return self._longitude

    def set_coords(self):
        try:
            lon, lat = self.from_file()
        except:
            sys.stdout.write('\nNo coordfiles ... calculating coords\n')
            lon, lat = self.from_array()
        self._latitude = lat
        self._longitude = lon

    '''
    Methods
    ~~~~~~~
    '''

    _files = {
        'g01_M2':   ('EASE2_M01km.lats.34704x14616x1.double', 'EASE2_M01km.lons.34704x14616x1.double'),
        'g03_M2':   ('EASE2_M03km.lats.11568x4872x1.double',  'EASE2_M03km.lons.11568x4872x1.double'),
        'g09_M2':   ('EASE2_M09km.lats.3856x1624x1.double',   'EASE2_M09km.lons.3856x1624x1.double'),
        'g125_N2':  ('EASE2_N12.5km.geolocation.v0.9.nc', ),
        'g25_N2':   ('EASE2_N25km.geolocation.v0.9.nc', ),
        'g03125_N2':('EASE2_N3.125km.geolocation.v0.9.nc', ),
        'g0625_N2': ('EASE2_N6.25km.geolocation.v0.9.nc', ),
        'g125_M2':  ('EASE2_T12.5km.geolocation.v0.9.nc', ),
        'g25_M2':   ('EASE2_T25km.geolocation.v0.9.nc', ),
        'g03125_M2':('EASE2_T3.125km.geolocation.v0.9.nc', ),
        'g0625_M2': ('EASE2_T6.25km.geolocation.v0.9.nc', ),

        'g12_M1': ('MHLATLSB', 'MHLONLSB'),
        'g25_M1': ('MLLATLSB', 'MLLONLSB'),

        'g12_N1': ('NHLATLSB', 'NHLONLSB'),
        'g25_N1': ('NLLATLSB', 'NLLONLSB'),
    }

    def from_file(self):
        ''' Not available for all files. '''

        file_key = '{}_{}'.format(self.grid_name, self.name)

        try:
            lat_file, lon_file = self._files[file_key]
        except ValueError as e:
            raise()

        if lat_file.endswith('.nc'):
            lat, lon = self._read_dataset(lat_file)
        elif lat_file.endswith('.double'):
            lat = self._read_flat(lat_file)
            lon = self._read_flat(lon_file)
        else:
            lat = self._read_coord(lat_file)
            lon = self._read_coord(lon_file)

        return lon, lat

    def _read_dataset(self, file_name):
        file_path = os.path.join(self._dat_path, file_name)
        with xr.open_dataset(file_path) as ds:
            return ds['latitude'].values, ds['latitude'].values

    def _read_flat(self, file_name):
        file_path = os.path.join(self._dat_path, file_name)
        dat_arr = np.fromfile(open(file_path, 'rb'), dtype='double')
        return dat_arr.reshape((self.constant.rows, self.constant.cols))

    def _read_coord(self, file_name):
        file_name = os.path.join(self._dat_path, file_name)
        coord = np.fromfile(file_name, dtype='i')
        coord.resize((self.constant.rows, self.constant.cols))
        return coord * self.coord_scale

    def from_array(self):
        """
        Calculate lat, lon from projection.
        :return: lon, lat
        """
        s_, r_ = np.meshgrid(np.arange(1, self.constant.cols + 1), np.arange(1, self.constant.rows + 1))
        return self.inverse(s_, r_)

    def forward(self, lon, lat, rowcol=True):
        '''Lat/lon forward projected to raster/scan.
        note: one-based index         '''
        x_, y_ = self.p(lon, lat)
        if not rowcol:
            return x_, y_
        assert self.units == 'm'
        s_, r_ = self.itransform * (x_, y_)
        return r_ + .5, s_ + .5

    def inverse(self, s_, r_):
        '''Raster/scan inverse projected to lat/lon.
        note: one-based index         '''
        assert self.units == 'm'
        # into meters from origin
        x_, y_ = self.transform * (s_ - .5, r_ - .5)
        lon, lat = self.p(x_, y_, inverse=True)
        return lon, lat

'''
Global Cylindrical
~~~~~~~~~~~~~~~~~~
'''

class M1(_Grid):
    # EASEGrid ver1 Global, equal area cylindrical projection.

    name = 'M1'
    hemi = 'M'
    proj = 'cea'
    lat_0 = 0
    lon_0 = 0
    lat_ts = 30
    a = 6371228.0
    units = 'm'
    p_args = ['proj', 'lat_0', 'lon_0', 'lat_ts', 'a', 'units']
    grid_names = _Grid._constant[name].keys()

class M2(_Grid):
    # EASEGrid ver2 Global, equal area cylindrical projection.

    name = 'M2'
    hemi = 'M'
    proj = 'cea'
    lat_0 = 0
    lon_0 = 0
    lat_ts = 30
    x_0 = 0
    y_0 = 0
    ellps = 'WGS84'
    datum = 'WGS84'
    units = 'm'
    p_args = ['proj', 'lat_0', 'lon_0', 'lat_ts', 'x_0', 'y_0', 'ellps', 'datum', 'units']
    grid_names = _Grid._constant[name].keys()

'''
Polar Azimuthal
~~~~~~~~~~~~~~~
'''

class P2(_Grid):
    # EASEGrid ver2 Polar, lambert azimuthal equal area

    name = 'P2'
    proj = 'laea'
    lon_0 = 0
    x_0 = 0
    y_0 = 0
    ellps = 'WGS84'
    datum = 'WGS84'
    units = 'm'
    p_args = ['proj', 'lat_0', 'lon_0', 'x_0', 'y_0', 'ellps', 'datum', 'units']
    grid_names = _Grid._constant[name].keys()


class S2(P2):
    lat_0 = -90
    hemi = 'S'


class N2(P2):
    lat_0 = 90
    hemi = 'N'


class P1(_Grid):
    # EASEGrid ver1 Polar, lambert azimuthal equal area

    name = 'P1'
    proj = 'laea'
    lon_0 = 0
    x_0 = 0
    y_0 = 0
    a = 6371228.0
    units = 'm'
    p_args = ['proj', 'lat_0', 'lon_0', 'x_0', 'y_0', 'a', 'units']
    grid_names = _Grid._constant[name].keys()


class S1(P1):
    lat_0 = -90
    hemi = 'S'


class N1(P1):
    lat_0 = 90
    hemi = 'N'