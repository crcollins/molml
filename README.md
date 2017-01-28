MolML
=====
[![Build Status](https://travis-ci.org/crcollins/molml.svg?branch=master)](https://travis-ci.org/crcollins/molml)
[![Coverage Status](https://coveralls.io/repos/github/crcollins/molml/badge.svg?branch=master)](https://coveralls.io/github/crcollins/molml?branch=master)
[![PyPI version](https://img.shields.io/pypi/v/MolML.svg?style=flat)](http://pypi.python.org/pypi/MolML)
[![License](https://img.shields.io/pypi/l/MolML.svg?style=flat)](https://github.com/crcollins/molml/blob/master/LICENSE.txt)


A library to interface molecules and machine learning. The goal of this library is to be a simple way to convert molecules into a vector representation for later use with libraries such as [scikit-learn](http://scikit-learn.org/). This is done using a similar API scheme.

All of the coordinates are assumed to be in angstroms.


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
    CoulombMatrix(input_type='list', n_jobs=1)
    >>> feat.transform([H2])
    array([[ 0.5,  1. ,  0. ,  1. ,  0.5,  0. ,  0. ,  0. ,  0. ]])
    >>> feat.transform([H2, HCN])
    array([[  0.5      ,   1.       ,   0.       ,   1.       ,   0.5      ,
            0.       ,   0.       ,   0.       ,   0.       ],
            [  0.5      ,   6.       ,   3.5      ,   6.       ,  36.8581052,
            42.       ,   3.5      ,  42.       ,  53.3587074]])
```

For more examples, look in the [examples](https://github.com/crcollins/molml/tree/master/examples). Note: To run some of the examples scikit-learn>=0.16.0 is required.


Dependencies
============

MolML works with both Python 2 and Python 3 and depends on numpy, scipy, and pathos (and future for Python 2). The specific versions that have been tested are Python 2.7/3.4/3.5, numpy 1.9.1, scipy 0.15.1, and pathos 0.2.0, but newer versions should work. 

NOTE: Due to an issue with multiprocess (a pathos dependency), the minimum version of python that will work is 2.7.4. For full details see [this link](https://github.com/uqfoundation/multiprocess/issues/11). Without this, the parallel computation of features will fail.


Install
=======

Once the dependencies are installed, the package can be installed with pip.

    $ pip install molml

Or for the bleeding edge version, you can use

    $ pip install git+git://github.com/crcollins/molml


Development
===========

To install a development version, just clone the git repo.

    $ git clone https://github.com/crcollins/molml
    $ # cd to molml and setup some virtualenv
    $ pip install -r requirements.txt
    $ pip install -r requirements-dev.txt

Pull requests and bug reports are welcomed!


Testing
=======

To run the tests, make sure that `nose` is installed and then run:

    $ nosetests

To include coverage information, make sure that `coverage` is installed and then run:

    $ nosetests --with-coverage --cover-package=molml --cover-erase

