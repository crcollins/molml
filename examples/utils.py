from __future__ import print_function
import os
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
from builtins import range

import numpy
import scipy.io
from scipy.constants import physical_constants, angstrom

from molml.utils import NUM_TO_ELE


BOHR_TO_ANGSTROMS = physical_constants['Bohr radius'][0] / angstrom


def download_data():
    '''
    Download the QM7 data set
    '''
    url = "http://quantum-machine.org/data/qm7.mat"
    print("Downloading data")
    response = urlopen(url)
    print("Writing data")
    with open("qm7.mat", "wb") as f:
        f.write(response.read())
    print("Data written")


def convert_input(Xin):
    '''
    Convert the QM7 data to the proper format

    This removes all the padding values from the elements and coords
    '''
    new = []
    for z, r in Xin:
        temp = [(NUM_TO_ELE[int(x)], y) for x, y in zip(z, r) if x]
        zs, rs = zip(*temp)
        new.append((numpy.array(zs), numpy.array(rs)))
    return new


def get_fold_idxs(P, fold=0):
    train_folds = [x for x in range(5) if x != fold]
    train_idxs = numpy.ravel(P[train_folds])
    test_idxs = numpy.ravel(P[fold])
    return train_idxs, test_idxs


def get_data_train_test(data, fold=0):
    train_idxs, test_idxs = get_fold_idxs(data['P'], fold=fold)

    y_train = data['T'][train_idxs]
    y_test = data['T'][test_idxs]

    R_train = data['R'][train_idxs]
    Z_train = data['Z'][train_idxs]
    Xin_train = list(zip(Z_train, R_train))

    R_test = data['R'][test_idxs]
    Z_test = data['Z'][test_idxs]
    Xin_test = list(zip(Z_test, R_test))
    return Xin_train, Xin_test, y_train, y_test


def load_qm7_data():
    '''
    Load the QM7 data set
    '''
    if not os.path.exists("qm7.mat"):
        download_data()
    data = scipy.io.loadmat("qm7.mat")

    R = BOHR_TO_ANGSTROMS * data['R']
    Xin = zip(data['Z'], R)
    Xout = convert_input(Xin)
    Z, R = zip(*Xout)
    new_data = {
        'P': data['P'],
        'R': numpy.array(R),
        'Z': numpy.array(Z),
        'T': numpy.ravel(data['T']),
    }
    return new_data


def load_qm7(fold=0):
    data = load_qm7_data()
    return get_data_train_test(data, fold=fold)
