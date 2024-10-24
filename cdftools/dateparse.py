#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
cdftools/dateparse.py
Tools for parsing dates
'''

from dateutil.parser import parse as du_parse
import arrow

# For python2/python3 support
try:
    basestring
except NameError:
    basestring = str

def dateparse(date_str):
    '''
    Returns a native datetime object at UTC time with timezone set, parsed from
    an ISO-8601 string.

    :param str date_str: Datetime string in ISO-8601
    '''
    if isinstance(date_str, basestring):
        if date_str.endswith('+00'):
            date_str = date_str.replace('+00', 'Z')
    date_obj = du_parse(date_str)
    arrow_obj = arrow.get(date_obj)
    return arrow_obj.to('utc').naive


