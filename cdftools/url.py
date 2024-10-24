#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
cdftools/url.py

Tools for manipulating urls
'''

from six.moves.urllib.parse import urlparse, parse_qs, urlunparse, urlencode


def get_url_query(url):
    '''
    Returns the query portion of a URL as a dictionary

    :param str url: A URL
    '''
    parts = urlparse(url)
    query = {k: v[0] for k, v in parse_qs(parts.query, True).items()}
    return query


def patch_url_query(url, query):
    '''
    Returns a patched URL by replacing the query portion with the query specified

    :param str url: URL to patch
    :param dict query: Query arguments
    '''
    parts = urlparse(url)
    url = urlunparse(parts._replace(query=urlencode(query)))
    return url

