#!/usr/bin/env python
'''
tests/test_plot_tools.py
'''
import matplotlib
matplotlib.use('Agg')
from unittest import TestCase
import datetime
import numpy as np
import matplotlib.pyplot as plt
from cdftools.plot_tools import OceanPlots


class TestPlotTools(TestCase):
    '''
    Tests the plotting functionality of cdftools. This
    merely checks that they plot without errors
    '''

    def setUp(self):
        self.ocean_plots = OceanPlots()

    def test_plot_timeseries(self):
        '''
        Ensure that you can plot a timeseries without error
        '''
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        numdays = 100
        base = datetime.datetime.today()
        date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
        y = np.random.random(len(date_list)) * 360
        self.ocean_plots.plot_time_series(fig, ax, date_list, y,
                                          title='test',
                                          ylabel="test")
        plt.close(fig)


    def test_plot_rose(self):
        '''
        Ensure that you can plot a rose plot without error
        '''

        # Create speed and direction variables
        N = 500
        magnitude = np.random.random(N) * 6
        direction = np.random.random(N) * 360
        fig, ax = self.ocean_plots.plot_rose(magnitude, direction,
                                             title='title',
                                             legend_title='legend_title')
        plt.close(fig)

    def test_skewTlogP(self):
        '''
        Ensure that you can plot a skewTlogP plot without error
        '''

        fig, ax = plt.subplots(1, 1, figsize=(7, 9))
        pressure = np.array([1011.1295166 ,  975.9017334 ,  904.42218018,  804.57861328])
        temperature = np.array([16.17812538, 13.14687538,  7.1624999 , 10.20937538])
        dewpoint = np.array([  9.57363319,   8.36019325,   6.75466299, -20.87717247])
        windspeed = np.array([17.4500935 , 19.54003852, 17.06053773, 10.60136506])
        winddir = np.array([93.33022692, 93.87385112, 93.29558601,  5.54267083])
        fig = self.ocean_plots.plot_skew_t_log_p(fig, pressure, temperature, dewpoint, windspeed, winddir, 'title')
        plt.close(fig)

