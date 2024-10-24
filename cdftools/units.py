#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
cdftools/units.py
'''
# C. Turner , Cathleen.Turner@rpsgroup.com 2/17/2016

from __future__ import print_function

import re
from cf_units import Unit


# The nomenclature and the standard units
# Example: If analysed_sst comes in with kelvin units
# and we want that mapped to temperature, we should
# be sure that is mapped here and in UNITS_NOMEN below.
SYSTEMS = {
    "temperature": "degrees_Celsius",
    "current_speed": "knots",
    "wind_speed": "knots",
    "direction": "degrees",
    "air_pressure": "mbar",
    "sea_pressure": "dbar",
    "latitude": "degrees_north",
    "longitude": "degrees_east",
    "relative_humidity": "percent",
    "wave_height": "m",
    "wave_period": "s",
    "conductivity": "S m-1",
    "probability": "percent",
    "density": "kg m-3",
    "depth": "m",
    "water_level": "m",
    "stream_flow": "m3/s",
    "precipitation": "mm",
    "radiation": "mmoles m-2",
    "mass_concentration_of_oxygen_in_sea_water": "mg L-1",
    "mass_concentration_of_chlorophyll_in_sea_water": "ug L-1",
    "analysed_sst": "kelvin",
    "sst": "kelvin",
    "visibility_in_air": "miles"
}

# All of the 'bad' units and their respective system (aka nomenclature)
# include correct spelling and weird spelling units here
# Example: If you wanted kelvin to be converted to temperature values
# such as deg_C and deg_F, a "temperature" value would be added to kelvin.
UNITS_NOMEN = {
    "deg": "direction",
    "hPa": "air_pressure",
    "Pa": "air_pressure",
    "mbar": "air_pressure|sea_pressure",
    "S/m": "conductivity",
    "psu": "conductivity",
    "knots": "current_speed|wind_speed",
    "C": "temperature",
    "Cel": "temperature",
    "deg_C": "temperature",
    "degree_Celsius": "temperature",
    "degrees_Celsius": "temperature",
    "deg_F": "temperature",
    "degF": "temperature",
    "F": "temperature",
    "degree": "temperature|direction|latitude|longitude",   # ex: degrees. For one source it means degrees F. For another it means degrees direction
    "degrees": "temperature|direction|latitude|longitude",
    "m/s": "current_speed|wind_speed",
    "cm/s": "current_speed",
    "meters s-1": "current_speed|wind_speed",
    "m s-1": "current_speed|wind_speed",
    "ft": "water_level|depth|wave_height|precipitation",
    "Ft": "water_level|depth|wave_height|precipitation",
    "meters": "water_level|depth|wave_height|precipitation",
    "m": "water_level|depth|wave_height|precipitation",
    "mm": "water_level|depth|wave_height|precipitation",
    "ft3/s": "stream_flow",
    "mS/cm": "conductivity",
    "mmoles m-2": "radiation",
    "mmoles/m^2": "radiation",
    "mmoles ft-2": "radiation",
    "kg m-3": "mass_concentration_of_oxygen_in_sea_water|mass_concentration_of_chlorophyll_in_sea_water|density",
    "mg L-1": "mass_concentration_of_oxygen_in_sea_water|mass_concentration_of_chlorophyll_in_sea_water",
    "kelvin": "temperature",
    "Kelvin": "temperature",
    "K": "temperature"
}

# Bad units: bad units not understood by cf_units that need to be mapped
# https://www.cicsnc.org/pub/jbiard/Udunits2Tables.html
UNITS2CONVERT = {
    "degF": "deg_F",
    "deg_F": "deg_F",
    "F": "deg_F",
    "Degrees in Celcius": 'deg_C',
    "degree_Celsius": "deg_C",
    "degrees_Celsius": 'deg_C',
    "degrees_C": "deg_C",
    "deg_c": "deg_C",
    "Celsius": 'deg_C',
    "celsius": 'deg_C',
    "deg C": "deg_C",
    "degC": "deg_C",
    "C": 'deg_C',
    "°C": "deg_C",
    "Degrees Celcius": "deg_C",
    "Degree Celcius": "deg_C",
    "Degree Celsius": "deg_C",
    "degrees_celsius": "deg_C",
    "feet": "ft",
    "Ft": "ft",
    "Hz": "hertz",
    "mb": "mbar",
    "Knots": "knots",
    "Degrees": "degrees",
    "degree.true": "degrees",
    "°TN": "degrees",
    "deg": "degrees",
    "degrees_true": "degrees",
    "degrees_north": "degrees",
    "degT": "degrees",
    "Meters": "meters",
    "m": "meters",
    "DirectionTo Angle in Degrees": "degrees",
    "DirectionTo Angles In Degrees": "degrees",
    "Meters Per Second": "m s-1",
    "millimeters": "mm",
    "m/s": "m s-1",
    "meters/second": "m s-1",
    "mmoles/m^2": "mmoles m-2",
    "nautical miles": "nmiles",
    "nmi": "nmiles",
    "percentage": "percent",
    "s": "second",
    "US_statute_miles": "miles"
}

# Regular Expression Lookup tables
# this has the dictionaries for the ambiguous units and corresponding variables
REG_EX_STRINGS = {
    "(Height of Combined Wind, Waves and Swells)": "wave_height",
    "(dir)": "direction",
    "(air_pressure)": "air_pressure",
    "(BP)": "air_pressure",
    "(temp|air)": "temperature",          # different expression can lead to reg_expression_temp
    "(Data Provider Y)": "temperature|deg_F",  # for special cases, may be needed later
    "(lat)": "latitude",
    "(lon)": "longitude",
    "(wind)": "wind_speed",
    "[wW]Spd": "wind_speed",
    "(current)": "current_speed",
    "(sea_water_speed)": "current_speed",
    "(depth)": "depth",
    "(water level)": "water_level",
    "(one_minute_water_level)": "water_level",
    "(elevation)": "water_level",
    "(Gage_height)": "water_level",
    "(Water Velocity)": "current_speed",
    "[wW]ave.*[hH]eight": "wave_height",
    "[hH]eight.*[wW]ave": "wave_height",
    "[wW]ater.*[hH]eight": "water_level",
    "[hH]eight.*[wW]ater": "water_level",
    "[sS]ea[ _][sS]urface[ _][hH]eight": "water_level",
    "[pP]rcp": "precipitation",
    "[rR]ain": "precipitation",
    "[dD]ensity": "density",
    "[cC]hlorophyll": "mass_concentration_of_chlorophyll_in_sea_water",
    "[oO]xygen": "mass_concentration_of_oxygen_in_sea_water"
}

# Units that are idenfitied as mislabled and can not be read by UDUnits should
# go here.
REPLACE_UNITS = {
    "Cel": "degrees_C"
}


def determine_units(units, value_input, var_str_in, system):
    '''
    Check the units and if necessary convert to the correct system

    :param str units: String describing the units
    :param number value_input: A valid number type
    :param str var_str_in: Name of the variable
    '''

    try:  # check if in dictionary of bad units or ambiguous units
        # Check the units input
        units = get_units(units)
        if units in UNITS2CONVERT:
            units = UNITS2CONVERT[units]
        # cf_name = UNITS2CONVERT[units]

        # If we don't specify an explicit mapping, and the units name has a
        # designated system, use the system.

        # For example, if you say "meters/second" with wind_speed, you'll get
        # knots.
        if units not in UNITS_NOMEN and var_str_in.lower() in system:
            nomen = var_str_in.lower()
        else:
            nomen = UNITS_NOMEN[units]

        if "|" in nomen:  # unit is ambiguous, use variable name to determine nomen
            results = convert_bad_variable(var_str_in, value_input, units, nomen, system)
        else:  # bad unit (unusual spelling) and not ambiguous
            try:  # executed if any conversions need to be done, and unit has spelling not recognized by cf units
                # if true (in dictionary) value needs to be converted known map of
                # bad_units:bad units understood by cf
                # cf_name = UNITS2CONVERT[units]
                changeto_cfunit = system[nomen]  # this is assuming that they are the same. may need dictionary of our_units->cf_units
                converted_value_out = convert_values(units, changeto_cfunit, value_input)
                results = {'values': converted_value_out,
                           'units': system[nomen],
                           'system': nomen}

            except:  # that means no conversion is necessary, just spelling of units is not recognized by cf units
                results = {'values': value_input,
                           'units': units,
                           'system': nomen}

    except Exception as e:
        # No matter what happens, we're returning SOMETHING
        # On error just return the inputs...
        results = {'values': value_input,
                   'units': units,
                   'system': var_str_in}
    finally:
        return results


def get_units(units):
    '''
    Returns a unit string (in case units is a list)
    '''
    if isinstance(units, list):
        if len(set(units)) > 1:  # Got passed in a list of units, not good!
            raise TypeError('Cant convert multiple units')
        units = units[0]

    if units in REPLACE_UNITS:
        units = REPLACE_UNITS[units]

    return units


def convert_values(units_in, units_out, values_in):
    '''
    Returns converted values or value from units_in to units_out.

    :param str units_in: Units to convert from
    :param str units_out: Units to convert to
    :param values_in: Values to convert
    '''
    if isinstance(values_in, list):
        values_out = []
        for value in values_in:
            if isinstance(value, list):
                values_inner = []
                for val in value:
                    values_inner.append(Unit(units_in).convert(val, units_out))
                values_out.append(values_inner)
            else:
                values_out.append(Unit(units_in).convert(value, units_out))
    else:
        values_out = Unit(units_in).convert(values_in, units_out)
    return values_out   # returns converted value


def convert_bad_variable(var_str_in, value_input, unit_in, nomen, system):
    '''
    Uses functions to get correct nomenclature and units from ambiguous units:

    :param str var_str_in: Name of the variable
    :param number value_input: Numeric value
    :param str unit_in: units for the variable
    :param str nomen: Nomenclature
    :param dict system: Dictionary describing rules for conversion
    '''
    actual_nomen = get_nomen_from_var_str(var_str_in, nomen)
    result_out = check_conv(actual_nomen, value_input, unit_in, system)
    return result_out


def get_nomen_from_var_str(var_str_in, nomen_in):
    '''
    Returns the actual nomenclature for a variable based on the variable name.

    Note: this method is used to deal with ambiguity in variables.

    :param str var_str_in: Name of the variable
    :param str nomen_in: Nomenclature
    '''
    for reg_expression, exp_name in REG_EX_STRINGS.items():
        regexp = re.compile(reg_expression, re.IGNORECASE)
        if regexp.search(var_str_in) and exp_name in nomen_in:
            actual_nomen = exp_name
            return actual_nomen


def check_conv(actual_nomen, value_input, unit_in, system):
    '''
    Converts the units using the provided nomenclature and units

    :param str actual_nomen: The actual nomenclature
    :param number value_input: Numeric type
    :param str unit_in: Units to convert from
    :param dict system: A dictionary of rules mapping nomenclature to units to convert to
    '''
    if "|" in actual_nomen:  # regex.find(actual_nomen) == -1: #that means some conversion needs to be done and units need to be converted
        split_nomen = actual_nomen.split("|")
        s_nomen = split_nomen[0]
        bad_units = split_nomen[1]
        change_to_cfunit = system[s_nomen]
        converted_value_out = convert_values(bad_units, change_to_cfunit, value_input)
        results = {'values': converted_value_out,
                   'units': system[s_nomen],
                   'system': s_nomen}
        return results
    else:  # we get a nomen with known nomenclature, thus known units
        change_to_cfunit = system[actual_nomen]
        cf_name = unit_in

        try:
            converted_value_out = convert_values(cf_name, change_to_cfunit, value_input)
        except:
            converted_value_out = value_input  # if units match nomenclature, but a conversion is simply not needed and cf units doesn't recognise the units (eg degrees)

        results = {'values': converted_value_out,
                   'units': change_to_cfunit,
                   'system': actual_nomen}
        return results


def convert_units(units, values, var_name, system=None):
    '''
    Returns a tuple of units and values converted into the appropriate units
    based on the system they belong to.

    If no system is specified, the default one is used (CF/IOOS).

    :param str units: The units the variable is in
    :param values: Numpy array or numeric types of values
    :param str var_name: Name of the variable as it appears in the source
    :param dict system: A dictionary mapping nomenclature to desired units
    '''
    global SYSTEMS
    system = system or SYSTEMS

    results = determine_units(units, values, var_name, system)
    return results['units'], results['values']
