#!/usr/bin/env python
'''
tests/test_units.py
'''

from unittest import TestCase
from cdftools.units import convert_units, determine_units, SYSTEMS
from copy import deepcopy
import numpy as np


class TestUnits(TestCase):
    '''
    Tests for converting units
    '''

    def setUp(self):
        self.default_system = deepcopy(SYSTEMS)

    def test_sea_surface_elevation_conversion(self):
        '''
        Direct test of the unit conversion method feet to meters
        '''
        units, val = convert_units('Ft', 6, 'surface elevation')
        # assert 'meters' in units
        assert 'm' in units
        assert abs(val - 1.8288) < 0.001

    def test_current_speed_conversion(self):
        '''
        Direct test of the unit conversion method m/s to knots
        '''
        units, val = convert_units('m/s', 0.5, 'current_speed')
        assert 'knots' in units
        assert abs(val - 0.971922) < 0.001

        units, val = convert_units('cm/s', 25, 'sea_water_speed')
        assert 'knots' in units
        assert abs(val - 0.48596) < 0.001

        # CBIBS current speed
        units, val = convert_units('mm/s', 25, 'Current_Speed')
        assert 'knots' in units
        assert abs(val - 0.048596) < 0.001

    def test_wind_speed_conversion(self):
        '''
        Direct test of the unit conversion method knots to m/s
        '''
        units, val = convert_units('m/s', 5.144444444444445, 'wind_speed')
        assert 'knots' in units
        np.testing.assert_allclose(val, 10.0)

    def test_pressure_conversion(self):
        '''
        Direct test of the unit conversion method hPa to mbar
        '''
        units, val = convert_units('Pa', 1000, 'air_pressure')
        assert 'mbar' in units
        assert abs(val - 10) < 0.001

    def test_kelvin_to_degrees(self):
        '''
        Direct test of the unit conversion kelvin to degrees
        '''
        units, val = convert_units('kelvin', 300, 'analysed_sst', SYSTEMS)
        units2, val2 = convert_units('Kelvin', 300, 'analysed_sst', SYSTEMS)
        units3, val3 = convert_units('K', 300, 'sst', SYSTEMS)

        assert 'degrees_Celsius' in units
        assert 'degrees_Celsius' in units2
        assert 'degrees_Celsius' in units3
        assert round(val, 2) == 26.85
        assert round(val2, 2) == 26.85
        assert round(val3, 2) == 26.85

    def test_ww3_wave_direction(self):
        '''
        Direct test of the bad ww3 units degree.true for wave direction
        '''
        units, val = convert_units('degree.true', 180, 'Primary_wave_direction_surface')
        assert 'degrees' in units
        assert val == 180

    def test_ndbc_wave_direction(self):
        '''
        Direct test of the bad ndbc units deg for wave direction
        '''
        units, val = convert_units('deg', 180, 'sea_surface_wave_to_direction')
        assert 'degrees' in units
        assert val == 180

    def test_wind_speed(self):
        units, val = convert_units('meters/second', 2, 'wind_speed')
        assert units == 'knots'
        np.testing.assert_allclose(val, 3.8876889848812093)

    def test_water_velocity_current_unit_conversion(self):
        units, val = convert_units('m/s', 1, 'Water Velocity', self.default_system)
        assert units == 'knots'
