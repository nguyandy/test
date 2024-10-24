#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
tests/test_units_system.py
'''

from unittest import TestCase
from cdftools.units import convert_units, SYSTEMS
from copy import deepcopy
import numpy as np


class TestUnitsSystem(TestCase):
    '''
    Tests for converting units
    '''

    def test_temperature(self):
        '''
        Tests system with temperature
        '''

        # Define US system
        system = deepcopy(SYSTEMS)
        system['temperature'] = 'degrees_F'

        units, val = convert_units('deg_C', 25.0, 'sea_water_temperature', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 77.0)

    def test_temperature_f(self):
        '''
        Tests that correct shorthand strings work
        '''
        system = deepcopy(SYSTEMS)
        system['temperature'] = 'degrees_C'
        units, val = convert_units('deg_F', 75.0, 'sea_water_temperature', system)
        assert units == 'degrees_C'
        np.testing.assert_almost_equal(val, 23.88888888888893)
        units, val = convert_units('F', 75.0, 'sea_water_temperature', system)
        assert units == 'degrees_C'
        np.testing.assert_almost_equal(val, 23.88888888888893)

    def test_dmme(self):
        '''
        Tests that the system in DMME works as a passed in system
        '''

        # Define US system
        system = deepcopy(SYSTEMS)
        system['wave_height'] = 'ft'
        system['water_level'] = 'ft'
        system['temperature'] = 'degrees_F'

        units, val = convert_units('m', 2.0, 'Height of Combined Wind, Waves and Swells', system)
        assert units == 'ft'
        np.testing.assert_almost_equal(val, 6.561679790026246)

        units, val = convert_units('m', 2.0, 'water level', system)
        assert units == 'ft'
        np.testing.assert_almost_equal(val, 6.561679790026246)

        units, val = convert_units('deg_C', 25.0, 'sea_water_temperature', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 77.0)

        units, val = convert_units('m', 2.0, 'Sea Surface Swell Wave Significant Height', system)
        assert units == 'ft'
        np.testing.assert_almost_equal(val, 6.561679790026246)

        units, val = convert_units('m', 2.0, 'sea_surface_height_amplitude_due_to_equilibrium_ocean_tide', system)
        assert units == 'ft'
        np.testing.assert_almost_equal(val, 6.561679790026246)

        units, val = convert_units('degree.true', 180.0, 'Primary_wave_direction_surface', system)
        assert units == 'degrees'
        np.testing.assert_almost_equal(val, 180.0)

        units, val = convert_units('deg', 180.0, 'sea_surface_wave_to_direction', system)
        assert units == 'degrees'
        np.testing.assert_almost_equal(val, 180.0)

    def test_eds_data(self):
        '''
        Tests that the data returned by EDS can be converted
        '''

        # Define US system
        system = deepcopy(SYSTEMS)
        system['wave_height'] = 'ft'
        system['water_level'] = 'ft'
        system['temperature'] = 'degrees_F'

        # Use the default system
        units, val = convert_units('Celsius', 17.4, 'sea_surface_temperature')
        assert units == 'degrees_Celsius'
        np.testing.assert_almost_equal(val, 17.4)

        units, val = convert_units(['Degrees in Celcius'], [17.4], 'Water Temperature')
        assert units == 'degrees_Celsius'
        np.testing.assert_almost_equal(val, 17.4)

        # Use the defined US system (from above)
        units, val = convert_units('Celsius', 17.4, 'sea_surface_temperature', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 63.32)

        units, val = convert_units(['Degrees in Celcius'], [17.4], 'Water Temperature', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 63.32)

    def test_coops_data(self):
        '''
        Tests that the data returned by COOPS can be converted
        '''

        # Define US system
        system = deepcopy(SYSTEMS)
        system['wave_height'] = 'ft'
        system['water_level'] = 'ft'
        system['temperature'] = 'degrees_F'
        system['air_pressure'] = 'psi'

        # Use the default system
        units, val = convert_units('mb', 1023.2, 'air_pressure')
        assert units == 'mbar'
        np.testing.assert_almost_equal(val, 1023.2)

        units, val = convert_units('C', 17.4, 'air_temperature')
        assert units == 'degrees_Celsius'
        np.testing.assert_almost_equal(val, 17.4)

        units, val = convert_units('C', 17.4, 'sea_water_temperature')
        assert units == 'degrees_Celsius'
        np.testing.assert_almost_equal(val, 17.4)

        # Use the defined US system (from above)
        units, val = convert_units('mb', 1023.2, 'air_pressure', system)
        assert units == 'psi'
        np.testing.assert_almost_equal(val, 14.8402613)

        units, val = convert_units('C', 17.4, 'air_temperature', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 63.32)

        units, val = convert_units('C', 17.4, 'sea_water_temperature', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 63.32)

    def test_usgs_data(self):
        '''
        Tests that the data returned by USGS can be converted
        '''

        # Define US system
        system = deepcopy(SYSTEMS)
        system['stream_flow'] = 'ft3/s'
        system['water_level'] = 'ft'

        # Use the default system
        units, val = convert_units('ft3/s', 4.3, 'Streamflow')
        assert units == 'm3/s'
        np.testing.assert_almost_equal(val, 0.1217624)

        units, val = convert_units('ft', 3.33, 'Gage_height')
        assert units == 'm'
        np.testing.assert_almost_equal(val, 1.014984)

        # Use the defined US system (from above)
        units, val = convert_units('ft3/s', 4.3, 'Streamflow', system)
        assert units == 'ft3/s'
        np.testing.assert_almost_equal(val, 4.3)

        units, val = convert_units('ft', 3.33, 'Gage_height', system)
        assert units == 'ft'
        np.testing.assert_almost_equal(val, 3.33)

    def test_nerrs_data(self):
        '''
        Tests that the data returned by NERRS can be converted
        '''

        # Define US system
        system = deepcopy(SYSTEMS)
        system['conductivity'] = 'S ft-1'
        system['water_level'] = 'ft'
        system['temperature'] = 'degrees_F'
        system['wind_speed'] = 'mi/hr'
        system['precipitation'] = 'in'
        system['air_pressure'] = 'psi'
        system['radiation'] = 'mmoles ft-2'
        system['density'] = 'slugs ft-3'

        # Use the default system
        units, val = convert_units('°C', 0, 'Temp')
        assert units == 'degrees_Celsius'
        np.testing.assert_almost_equal(val, 0)

        units, val = convert_units('°C', 0, 'ATemp')
        assert units == 'degrees_Celsius'
        np.testing.assert_almost_equal(val, 0)

        units, val = convert_units('mS/cm', 47.12, 'SpCond')
        assert units == 'S m-1'
        np.testing.assert_almost_equal(val, 4.712)

        units, val = convert_units('m/s', 5.8, 'MaxWSpd')
        assert units == 'knots'
        np.testing.assert_almost_equal(val, 11.27429805)

        units, val = convert_units('m/s', 5.8, 'WSpd')
        assert units == 'knots'
        np.testing.assert_almost_equal(val, 11.27429805)

        units, val = convert_units('°TN', 177.0, 'Wdir')
        assert units == 'degrees'
        np.testing.assert_almost_equal(val, 177.0)

        units, val = convert_units('mm', 1.0, 'TotPrcp')
        assert units == 'mm'
        np.testing.assert_almost_equal(val, 1.0)

        units, val = convert_units('mb', 1023.2, 'BP')
        assert units == 'mbar'
        np.testing.assert_almost_equal(val, 1023.2)

        units, val = convert_units('mmoles/m^2', 578.0, 'TotPAR')
        assert units == 'mmoles m-2'
        np.testing.assert_almost_equal(val, 578.0)

        units, val = convert_units('mg/L', 10.33, 'mass_concentration_of_oxygen_in_sea_water')
        assert units == 'mg L-1'
        np.testing.assert_almost_equal(val, 10.33)

        units, val = convert_units('kg m-3', 1024, 'sea_water_density')
        assert units == 'kg m-3'
        np.testing.assert_almost_equal(val, 1024)

        # Use the defined US system (from above)
        units, val = convert_units('°C', 0, 'Temp', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 32)

        units, val = convert_units('°C', 0, 'ATemp', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 32)

        units, val = convert_units('mS/cm', 47.12, 'SpCond', system)
        assert units == 'S ft-1'
        np.testing.assert_almost_equal(val, 1.4362176)

        units, val = convert_units('m/s', 5.8, 'MaxWSpd', system)
        assert units == 'mi/hr'
        np.testing.assert_almost_equal(val, 12.974230493)

        units, val = convert_units('m/s', 5.8, 'WSpd', system)
        assert units == 'mi/hr'
        np.testing.assert_almost_equal(val, 12.974230493)

        units, val = convert_units('mm', 1.0, 'TotPrcp', system)
        assert units == 'in'
        np.testing.assert_almost_equal(val, 0.0393701)

        units, val = convert_units('mb', 1023.2, 'BP', system)
        assert units == 'psi'
        np.testing.assert_almost_equal(val, 14.8402613)

        units, val = convert_units('mmoles/m^2', 578.0, 'TotPAR', system)
        assert units == 'mmoles ft-2'
        np.testing.assert_almost_equal(val, 53.69795712)

        units, val = convert_units('kg m-3', 1024, 'sea_water_density', system)
        assert units == 'slugs ft-3'
        np.testing.assert_almost_equal(val, 1.986888419)

    def test_hrecos_data(self):
        '''
        Tests that the data returned by HRECOS can be converted
        '''

        # Define US system
        system = deepcopy(SYSTEMS)
        system['precipitation'] = 'in'

        # Use the default system
        units, val = convert_units('m', 0.2, 'thickness_of_rainfall_amount')
        assert units == 'mm'
        np.testing.assert_almost_equal(val, 200)

        units, val = convert_units('kg m-3', 0.012419, 'mass_concentration_of_oxygen_in_sea_water')
        assert units == 'mg L-1'
        np.testing.assert_almost_equal(val, 12.419)

        # Use the defined US system (from above)
        units, val = convert_units('m', 0.2, 'thickness_of_rainfall_amount', system)
        assert units == 'in'
        np.testing.assert_almost_equal(val, 7.874015748)

    def test_cbibs_data(self):
        '''
        Tests that the data returned by CBIBS can be converted
        '''

        # Define US system
        system = deepcopy(SYSTEMS)
        system['precipitation'] = 'in'
        system['conductivity'] = 'S ft-1'
        system['water_level'] = 'ft'
        system['temperature'] = 'degrees_F'
        system['wind_speed'] = 'mi/hr'
        system['precipitation'] = 'in'
        system['air_pressure'] = 'psi'
        system['radiation'] = 'mmoles ft-2'

        # Use the default system
        units, val = convert_units('C', 0, 'heat_index')
        assert units == 'degrees_Celsius'
        np.testing.assert_almost_equal(val, 0)

        units, val = convert_units('C', 0, 'wind_chill')
        assert units == 'degrees_Celsius'
        np.testing.assert_almost_equal(val, 0)

        units, val = convert_units('S/m', 47.12, 'sea_water_electrical_conductivity')
        assert units == 'S m-1'
        np.testing.assert_almost_equal(val, 47.12)

        # Use the defined US system (from above)
        units, val = convert_units('C', 0, 'heat_index', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 32)

        units, val = convert_units('C', 0, 'wind_chill', system)
        assert units == 'degrees_F'
        np.testing.assert_almost_equal(val, 32)

        units, val = convert_units('S/m', 47.12, 'sea_water_electrical_conductivity', system)
        assert units == 'S ft-1'
        np.testing.assert_almost_equal(val, 14.362176)
