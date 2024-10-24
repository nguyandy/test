#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
cdftools/climate.py
Utilities for processing climatology data for netcdf files
'''

from __future__ import print_function
from six.moves import range
from netCDF4 import Dataset
from netCDF4 import num2date, date2index
import numpy as np
import calendar
from datetime import datetime
import logging

MISSING_VAL = None

logger = None


def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger(__name__)
    return logger


def get_unmasked(array):
    '''
    Replace fill values with NaNs
    '''
    if hasattr(array, 'mask'):
        data = array.data
        data[array.mask] = np.nan
        return data.tolist()
    return array.tolist()


def get_climatology_stats(url, provider, parameter, theme):
    '''
    Main entry function to get stats

    :param str url: url
    :param str provider: provider
    :param str parameter: parameter to use
    :param str theme: theme
    '''
    trend_data = None

    with Dataset(url, 'r') as nc:
        nc_time = nc.variables['time']

        if parameter in nc.variables:
            data = nc.variables[parameter]

            dates = num2date(nc_time[:], units=nc_time.units, calendar=nc_time.calendar)

            start_year = dates[0].year
            end_year = dates[-1].year

            values_monthly = calculate_monthly(start_year, end_year, data, dates, nc_time)
            values_annual = calculate_monthly_all_years(start_year, end_year, data, dates, nc_time)

            units = ''
            if 'units' in data.ncattrs():
                units = data.getncattr('units')

            short_name = parameter
            if 'short_name' in data.ncattrs():
                short_name = data.getncattr('short_name')

            shape = len(data.shape)

            parameter_dict = {'units': [units],
                              'values_monthly': values_monthly,
                              'values_annual': values_annual}

            if shape == 2 and 'depth' in nc.variables:
                parameter_dict["depth"] = nc.variables['depth'][:].tolist()
            else:

                # create file for trend
                trend_data = {"geometry": {"coordinates": "", "type": "Point"},
                              "properties": {"data": {parameter: parameter_dict},
                                             "short_name": [short_name, "date_time"]},
                              "type": "Feature"
                              }

    return trend_data


def process_climate_monthly_all_years(data, start_year, end_year, month, dates, nc_time, index=None):
        '''
        This is the main portion of the analysis
        '''
        # Init output
        monthly_data = np.array([])
        monthly_dates = np.array([])
        monthly_means = np.array([])

        keys = ['min', 'min_date', 'max', 'max_date', 'mean', 'mean_range', 'npts', 'var', 'std']
        stats = {key: MISSING_VAL for key in keys}
        stats['npts'] = 0

        # Do the work
        for year in range(start_year, end_year + 1):
            start_day = 1
            end_day = calendar.monthrange(year, month)[1]
            try:
                # find first data after beginning of month
                month_start_date = datetime(year, month, start_day)
                start_month_idx = date2index(month_start_date, nc_time, calendar=nc_time.calendar, select='after')
                # find last data before end of month
                month_end_date = datetime(year, month, end_day, 23, 59, 59)
                end_month_idx = date2index(month_end_date, nc_time, calendar=nc_time.calendar, select='before')

                if start_month_idx < end_month_idx:
                    if index is not None:
                        month_data = data[start_month_idx:end_month_idx, index]
                    else:
                        month_data = data[start_month_idx:end_month_idx]

                    # Remove NaN's and FillValues for analysis
                    month_data = get_unmasked(month_data)
                    if len(month_data) > 0:
                        monthly_data = np.concatenate((monthly_data, month_data), axis=0)
                        month_dates = dates[start_month_idx:end_month_idx]
                        monthly_dates = np.concatenate((monthly_dates, month_dates), axis=0)
                        monthly_means = np.append(monthly_means, np.mean(month_data))
                    else:
                        raise ValueError('Data is all NaN')
                else:
                    raise ValueError('No Data found')

            except Exception as e:
                # Value Error here means theres no monthly data for that particular year
                # Can just just skip onto the next year
                logger = get_logger()
                logger.debug("Error getting climate data: %s" % e)

        npts = len(np.array(monthly_data)[~np.isnan(np.array(monthly_data))])
        if npts > 0:
            stats = {'min': np.nanmin(monthly_data),
                     'min_date': monthly_dates[np.argmin(monthly_data)].strftime('%Y-%m-%dT%H:%M:%SZ'),
                     'max': np.nanmax(monthly_data),
                     'max_date': monthly_dates[np.argmax(monthly_data)].strftime('%Y-%m-%dT%H:%M:%SZ'),
                     'mean': np.nanmean(monthly_data),
                     'mean_range': [np.nanmin(monthly_means), np.nanmax(monthly_means)],
                     'npts': npts,
                     'var': np.nanvar(monthly_data),
                     'std': np.nanstd(monthly_data)}

            stats = verify_data_fields(stats)

        return stats


def verify_data_fields(stats):
    '''
    verifies that the data is not nan and converts to None for json output
    '''
    for field in list(stats.keys()):
        if field == "mean_range":
            if np.isnan(stats[field][0]) or np.isnan(stats[field][1]):
                stats[field] = [None, None]
        else:
            if not isinstance(stats[field], str) and np.isnan(stats[field]):
                stats[field] = None
    return stats


def calculate_monthly_all_years(start_year, end_year, data, dates, nc_time):
    '''
    Gets monthly statistics for all available years (annual)

    :param int start_year:start_year
    :param int end_year:end_year
    :param dict data:data
    :param list dates:dates
    :param list nc_time:nc_time
    '''

    # Initialize data dictionary
    data_container = {}

    # Check for 2D data!
    shape = len(data.shape)

    # loop through the months
    # ("PROCESSING MONTHLY STATS (ALL YEARS)")
    for month in range(1, 13):
        # Initialize individual data components
        npts = mean_month = max_month = max_date = min_month = min_date = var = std = np.array([])
        mean_range = []
        if shape == 1:
            stats = process_climate_monthly_all_years(data, start_year, end_year, month, dates, nc_time, index=None)
            npts = np.append(npts, stats['npts'])
            mean_month = np.append(mean_month, stats['mean'])
            mean_range.append(stats['mean_range'])
            max_month = np.append(max_month, stats['max'])
            max_date = np.append(max_date, stats['max_date'])
            min_month = np.append(min_month, stats['min'])
            min_date = np.append(min_date, stats['min_date'])
            var = np.append(var, stats['var'])
            std = np.append(std, stats['std'])
        elif shape == 2:
            depths = []
            # This must mean we are working with 2D depth data
            # Put all the results into arrays
            for depth_ind in range(0, data.shape[1]):
                stats = process_climate_monthly_all_years(data, start_year, end_year, month, dates, nc_time, index=depth_ind)
                npts = np.append(npts, stats['npts'])
                mean_month = np.append(mean_month, stats['mean'])
                mean_range.append(stats['mean_range'])
                max_month = np.append(max_month, stats['max'])
                max_date = np.append(max_date, stats['max_date'])
                min_month = np.append(min_month, stats['min'])
                min_date = np.append(min_date, stats['min_date'])
                var = np.append(var, stats['var'])
                std = np.append(std, stats['std'])
                # depth index
                depths.append(depth_ind)
        else:
            # ("Data with " + str(shape) + " dimensions not supported")
            return {}

        data_container[str(month)] = {'min': min_month.tolist(),
                                      'min_date': min_date.tolist(),
                                      'max': max_month.tolist(),
                                      'max_date': max_date.tolist(),
                                      'mean': mean_month.tolist(),
                                      'mean_range': mean_range,
                                      'npts': npts.tolist(),
                                      'var': var.tolist(),
                                      'std': std.tolist()}

    return data_container


def process_climate_monthly(data, year, month, dates, nc_time, index=None):
        '''
        This is the main portion of the analysis
        '''
        keys = ['min', 'min_date', 'max', 'max_date', 'mean', 'npts', 'var', 'std']
        stats = {key: MISSING_VAL for key in keys}
        stats['npts'] = 0

        start_day = 1
        end_day = calendar.monthrange(year, month)[1]
        try:
            # find data after beginning of month
            month_start_date = datetime(year, month, start_day)
            start_month_idx = date2index(month_start_date, nc_time, calendar=nc_time.calendar, select='after')

            # find data before end of month
            month_end_date = datetime(year, month, end_day, 23, 59, 59)
            end_month_idx = date2index(month_end_date, nc_time, calendar=nc_time.calendar, select='before')
            # find data between month day ranges
            # check for issues with dates and need 10 points to do the analysis

            if (start_month_idx < end_month_idx) and ((end_month_idx - start_month_idx) > 10):
                if index is not None:
                    month_data = data[start_month_idx:end_month_idx, index]
                else:
                    month_data = data[start_month_idx:end_month_idx]

                # Remove NaN's and FillValue for analysis
                month_data = get_unmasked(month_data)
                if len(month_data) > 0:
                    npts = len(np.array(month_data)[~np.isnan(np.array(month_data))])
                    month_dates = dates[start_month_idx:end_month_idx]
                    # Get statistics of monthly data
                    stats = {'min': np.nanmin(month_data),
                             'min_date': month_dates[np.argmin(month_data)].strftime('%Y-%m-%dT%H:%M:%SZ'),
                             'max': np.nanmax(month_data),
                             'max_date': month_dates[np.argmax(month_data)].strftime('%Y-%m-%dT%H:%M:%SZ'),
                             'mean': np.nanmean(month_data),
                             'npts': npts,
                             'var': np.nanvar(month_data),
                             'std': np.nanstd(month_data)}

                    stats = verify_data_fields(stats)

                else:
                    raise ValueError('Month Data is all NaN')

            else:
                raise ValueError('No Data found')

        except Exception as e:
            # ('Skipping ' + str(year) + '/' + str(month) + ' monthly: ' + str(e))
            logger = get_logger()
            logger.debug("Error getting climate data, skipping %s %s: %s" % (year, month, e))

        return stats


def calculate_monthly(start_year, end_year, data, dates, nc_time):
    '''
    Calculates monthly statistics (trend)
    :param int start_year:start_year
    :param int end_year:end_year
    :param dict data:data
    :param list dates:dates
    :param list nc_time:nc_time
    '''

    # Initialize the dictionary
    monthly_data = {}
    # ("PROCESSING MONTHLY STATS")

    # Check for 2D data!
    shape = len(data.shape)
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Initialize individual data components
            npts = mean_month = max_month = max_date = min_month = min_date = var = std = np.array([])
            if shape == 1:
                stats = process_climate_monthly(data, year, month, dates, nc_time, index=None)
                npts = np.append(npts, stats['npts'])
                mean_month = np.append(mean_month, stats['mean'])
                max_month = np.append(max_month, stats['max'])
                max_date = np.append(max_date, stats['max_date'])
                min_month = np.append(min_month, stats['min'])
                min_date = np.append(min_date, stats['min_date'])
                var = np.append(var, stats['var'])
                std = np.append(std, stats['std'])
            elif shape == 2:
                depths = []
                # This must mean we are working with 2D depth data
                # Put all the results into arrays
                for depth_ind in range(0, data.shape[1]):
                    stats = process_climate_monthly(data, year, month, dates, nc_time, index=depth_ind)
                    npts = np.append(npts, stats['npts'])
                    mean_month = np.append(mean_month, stats['mean'])
                    max_month = np.append(max_month, stats['max'])
                    max_date = np.append(max_date, stats['max_date'])
                    min_month = np.append(min_month, stats['min'])
                    min_date = np.append(min_date, stats['min_date'])
                    var = np.append(var, stats['var'])
                    std = np.append(std, stats['std'])
                    # depth index
                    depths.append(depth_ind)
            else:
                # ("Data with " + str(shape) + " dimensions not supported")
                return {}

            if (str(year) not in monthly_data):
                monthly_data[str(year)] = {}
            if (str(month) not in monthly_data[str(year)]):
                monthly_data[str(year)][str(month)] = {'min': min_month.tolist(),
                                                       'min_date': min_date.tolist(),
                                                       'max': max_month.tolist(),
                                                       'max_date': max_date.tolist(),
                                                       'mean': mean_month.tolist(),
                                                       'npts': npts.tolist(),
                                                       'var': var.tolist(),
                                                       'std': std.tolist()}

    return monthly_data
