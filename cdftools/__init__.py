import logging


__version__ = '0.0.30'


LOGGER = None


def get_logger():
    '''
    Returns an initialized logger
    '''
    global LOGGER
    if LOGGER is None:
        LOGGER = logging.getLogger(__name__)
    return LOGGER


