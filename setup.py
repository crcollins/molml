#!/usr/bin/env python

# http://stackoverflow.com/questions/9810603/adding-install-requires-to-setup-py-when-making-a-python-package
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    import pypandoc
    try:
        LONG_DESCRIPTION = pypandoc.convert('README.md', 'rst')
    except:
        # Catch all exceptions because FileNotFoundError is only in 3.x
        from pypandoc.pandoc_download import download_pandoc
        download_pandoc()
        LONG_DESCRIPTION = pypandoc.convert('README.md', 'rst')
except ImportError:
    with open('README.md', 'r') as f:
        LONG_DESCRIPTION = f.read()

setup(
    name='molml',
    version='0.8.0',
    description='An interface between molecules and machine learning',
    long_description=LONG_DESCRIPTION,
    author='Chris Collins',
    author_email='chris@crcollins.com',
    url='https://github.com/crcollins/molml/',
    license='MIT',
    packages=['molml'],
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=[
        'pathos',
        'future',
    ],
    classifiers=[
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: POSIX",
                    "Programming Language :: Python :: 2",
                    "Programming Language :: Python :: 2.7",
                    "Programming Language :: Python :: 3",
                    "Programming Language :: Python :: 3.4",
                    "Programming Language :: Python :: 3.5",
                    "Programming Language :: Python :: 3.6",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Bio-Informatics",
                    "Topic :: Scientific/Engineering :: Chemistry",
                    "Topic :: Scientific/Engineering :: Physics",
    ]
)
