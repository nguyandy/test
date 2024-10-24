import os
import subprocess


def get_filename(path):
    '''
    Returns the path to a valid dataset
    '''
    filename = path
    if not os.path.exists(path):
        cdl_path = filename.replace('.nc', '.cdl')
        generate_dataset(cdl_path, filename)
    return filename


def generate_dataset(cdl_path, nc_path):
    subprocess.call(['ncgen', '-o', nc_path, cdl_path])


STATIC_FILES = {
    'point': get_filename('tests/data/point.nc'),
    'timeseries': get_filename('tests/data/timeseries.nc'),
    'multi-timeseries-orthogonal': get_filename('tests/data/multi-timeseries-orthogonal.nc'),
    'multi-timeseries-incomplete': get_filename('tests/data/multi-timeseries-incomplete.nc'),
    'trajectory': get_filename('tests/data/trajectory.nc'),
    'trajectory-single': get_filename('tests/data/trajectory-single.nc'),
    'profile-orthogonal': get_filename('tests/data/profile-orthogonal.nc'),
    'profile-incomplete': get_filename('tests/data/profile-incomplete.nc'),
    'timeseries-profile-single-station': get_filename('tests/data/timeseries-profile-single-station.nc'),
    'timeseries-profile-multi-station': get_filename('tests/data/timeseries-profile-multi-station.nc'),
    'timeseries-profile-single-ortho-time': get_filename('tests/data/timeseries-profile-single-ortho-time.nc'),
    'timeseries-profile-multi-ortho-time': get_filename('tests/data/timeseries-profile-multi-ortho-time.nc'),
    'timeseries-profile-ortho-depth': get_filename('tests/data/timeseries-profile-ortho-depth.nc'),
    'timeseries-profile-incomplete': get_filename('tests/data/timeseries-profile-incomplete.nc'),
    'trajectory-profile-orthogonal': get_filename('tests/data/trajectory-profile-orthogonal.nc'),
    'trajectory-profile-incomplete': get_filename('tests/data/trajectory-profile-incomplete.nc'),
    '2d-regular-grid': get_filename('tests/data/2d-regular-grid.nc'),
    '3d-regular-grid': get_filename('tests/data/3d-regular-grid.nc'),
}
