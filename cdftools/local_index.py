#!/usr/bin/env python
'''
cdftools/local_index.py

This module contains methods and utilities for creating a local index and aggregation of netCDF files
'''
from rtree.index import Rtree
from netCDF4 import Dataset
from cdftools import dap
from cdftools import get_logger
import arrow


class LocalIndex(object):
    def __init__(self, time_index_path, spatial_index_path):
        self.time_index = Rtree(time_index_path)
        self.spatial_index = Rtree(spatial_index_path)

    def build_index(self, paths):
        '''
        '''
        for path in paths:
            try:
                self.index_nc(path)
            except:
                get_logger().exception("Failed to index: %s", path)

    def index_nc(self, path):
        with Dataset(path, 'r') as nc:
            start_date, end_date = dap.get_time_extents(nc)
            t0 = arrow.get(start_date).timestamp
            t1 = arrow.get(end_date).timestamp
            bbox = dap.get_geo_extents(nc)
            bbox = (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])

            self.time_index.insert(hash(path), (t0, t1), obj=path)
            self.spatial_index.insert(hash(path), bbox, obj=path)

    def get_nearest(self, dt):
        '''
        Returns the URL of the nc file nearest the date

        :param datetime dt: Datetime of interest
        '''
        ts = arrow.get(dt).timestamp
        nearest = next(self.time_index.nearest((ts, ts), objects=True))
        return nearest.object
