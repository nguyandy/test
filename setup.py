from setuptools import setup, find_packages
from cdftools import __version__


def readme():
    with open('README.md') as f:
        return f.read()

reqs = [line.strip() for line in open('requirements.txt')]

setup(
    name                 = "cdftools",
    version              = __version__,
    description          = "A common python package containing tools for accessing netCDF files",
    long_description     = readme(),
    license              = 'Apache License 2.0',
    author               = "Luke Campbell",
    author_email         = "luke.campbell@rpsgroup.com",
    url                  = "https://github.com/asascience/cdftools",
    packages             = find_packages(exclude=['tests','tests.*']),
    install_requires     = reqs,
    tests_require        = ['pytest'],
    classifiers          = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
)
