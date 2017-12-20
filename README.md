MolML
=====
[![Build Status](https://travis-ci.org/crcollins/molml.svg?branch=master)](https://travis-ci.org/crcollins/molml)
[![Coverage Status](https://coveralls.io/repos/github/crcollins/molml/badge.svg?branch=master)](https://coveralls.io/github/crcollins/molml?branch=master)
[![Documentation Status](https://readthedocs.org/projects/molml/badge/?version=latest)](http://molml.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/MolML.svg?style=flat)](http://pypi.python.org/pypi/MolML)
[![License](https://img.shields.io/pypi/l/MolML.svg?style=flat)](https://github.com/crcollins/molml/blob/master/LICENSE.txt)

A library to interface molecules and machine learning. The goal of this library is to be a simple way to convert molecules into a vector representation for later use with libraries such as [scikit-learn](http://scikit-learn.org/). This is done using a similar API scheme.

All of the coordinates are assumed to be in angstroms.


Features
========


    - Simple interface to many common molecular descriptors and their variants
        - Molecule
            - Coulomb Matrix
            - Bag of Bonds
            - Encoded Bonds
            - Encoded Angles
            - Connectivity Counts
            - Autocorrelation
        - Atom
            - Shell
            - Local Encoded Bonds
            - Local Encoded Angles
            - Local Coulomb Matrix
            - Behler-Parrinello
        - Kernel
            - Atom Kernel
        - Crystal
            - Generallized Crystal
            - Ewald Sum Matrix
            - Sine Matrix
    - Parallel feature generation
    - Ability to save/load fit models
    - Multiple input formats supported (and ability to define your own)
    - Supports both Python 2 and Python 3


Example Usage
=============

```python
    >>> from molml.features import CoulombMatrix
    >>> feat = CoulombMatrix()
    >>> H2 = (
    ...         ['H', 'H'],
    ...         [
    ...             [0.0, 0.0, 0.0],
    ...             [1.0, 0.0, 0.0],
    ...         ]
    ... )
    >>> HCN = (
    ...         ['H', 'C', 'N'],
    ...         [
    ...             [-1.0, 0.0, 0.0],
    ...             [ 0.0, 0.0, 0.0],
    ...             [ 1.0, 0.0, 0.0],
    ...         ]
    ... )
    >>> feat.fit([H2, HCN])
    CoulombMatrix(input_type='list', n_jobs=1, sort=False, eigen=False, drop_values=False)
    >>> feat.transform([H2])
    array([[ 0.5,  1. ,  0. ,  1. ,  0.5,  0. ,  0. ,  0. ,  0. ]])
    >>> feat.transform([H2, HCN])
    array([[  0.5      ,   1.       ,   0.       ,   1.       ,   0.5      ,
            0.       ,   0.       ,   0.       ,   0.       ],
            [  0.5      ,   6.       ,   3.5      ,   6.       ,  36.8581052,
            42.       ,   3.5      ,  42.       ,  53.3587074]])
    >>>
	>>> # Example loading from files directly
    >>> feat2 = CoulombMatrix(input_type='filename')
    CoulombMatrix(input_type='filename', n_jobs=1, sort=False, eigen=False, drop_values=False)
    >>> paths = ['data/qm7/qm-%04d.out' % i for i in xrange(2)]
	>>> feat2.fit_transform(paths)
	array([[ 36.8581052 ,   5.49459021,   5.49462885,   5.4945    ,
			5.49031286,   0.        ,   0.        ,   0.        ,
			5.49459021,   0.5       ,   0.56071947,   0.56071656,
			0.56064037,   0.        ,   0.        ,   0.        ,
			5.49462885,   0.56071947,   0.5       ,   0.56071752,
			0.56064089,   0.        ,   0.        ,   0.        ,
			5.4945    ,   0.56071656,   0.56071752,   0.5       ,
			0.56063783,   0.        ,   0.        ,   0.        ,
			5.49031286,   0.56064037,   0.56064089,   0.56063783,
			0.5       ,   0.        ,   0.        ,   0.        ,
			0.        ,   0.        ,   0.        ,   0.        ,
			0.        ,   0.        ,   0.        ,   0.        ,
			0.        ,   0.        ,   0.        ,   0.        ,
			0.        ,   0.        ,   0.        ,   0.        ,
			0.        ,   0.        ,   0.        ,   0.        ,
			0.        ,   0.        ,   0.        ,   0.        ],
		[ 36.8581052 ,  23.81043959,   5.48396427,   5.48394941,
			5.4837656 ,   2.78378686,   2.78375582,   2.78376439,
			23.81043959,  36.8581052 ,   2.78378953,   2.78375777,
			2.78375823,   5.4839846 ,   5.48393324,   5.48376877,
			5.48396427,   2.78378953,   0.5       ,   0.56363019,
			0.56362464,   0.40019757,   0.39971446,   0.3261774 ,
			5.48394941,   2.78375777,   0.56363019,   0.5       ,
			0.56362305,   0.39971429,   0.32617621,   0.40019524,
			5.4837656 ,   2.78375823,   0.56362464,   0.56362305,
			0.5       ,   0.32617702,   0.40019469,   0.3997145 ,
			2.78378686,   5.4839846 ,   0.40019757,   0.39971429,
			0.32617702,   0.5       ,   0.56362996,   0.56362587,
			2.78375582,   5.48393324,   0.39971446,   0.32617621,
			0.40019469,   0.56362996,   0.5       ,   0.56362278,
			2.78376439,   5.48376877,   0.3261774 ,   0.40019524,
			0.3997145 ,   0.56362587,   0.56362278,   0.5       ]])
```

For more examples, look in the [examples](https://github.com/crcollins/molml/tree/master/examples). Note: To run some of the examples scikit-learn>=0.16.0 is required.

For the full documentation, refer to the [docs](http://molml.readthedocs.io) or the docstrings in the code.


Dependencies
============

MolML works with both Python 2 and Python 3. It has been tested with the versions listed below, but newer versions should work.

    python>=2.7/3.5/3.6
    numpy>=1.9.1
    scipy>=0.15.1
    pathos>=0.2.0
    future  # For python 2


NOTE: Due to an issue with multiprocess (a pathos dependency), the minimum version of Python that will work is 2.7.4. For full details see [this link](https://github.com/uqfoundation/multiprocess/issues/11). Without this, the parallel computation of features will fail.


Install
=======

Once `numpy` and `scipy` are installed, the package can be installed with pip.

    $ pip install molml

Or for the bleeding edge version, you can use

    $ pip install git+git://github.com/crcollins/molml


Development
===========

To install a development version, just clone the git repo.

    $ git clone https://github.com/crcollins/molml
    $ # cd to molml and setup some virtualenv
    $ pip install -r requirements-dev.txt

[Pull requests](https://github.com/crcollins/molml/pulls) and [bug reports](https://github.com/crcollins/molml/issues) are welcomed!

To build the documentation, you just need to install the documentation dependencies. These are already included in the dev install.

    $ cd docs/
    $ pip install -r requirements-docs.txt
    $ make html

Testing
=======

To run the tests, make sure that `nose` is installed and then run:

    $ nosetests

To include coverage information, make sure that `coverage` is installed and then run:

    $ nosetests --with-coverage --cover-package=molml --cover-erase

