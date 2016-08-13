import os

import numpy
import scipy.io
from sklearn.linear_model import Ridge

from molml.features import EncodedBond, BagOfBonds, Connectivity, CoulombMatrix
from molml.utils import NUM_TO_ELE


def download_data():
    import urllib2
    url = "http://quantum-machine.org/data/qm7.mat"
    print "Downloading data"
    response = urllib2.urlopen(url)
    print "Writing data"
    with open("qm7.mat", "w") as f:
        f.write(response.read())
    print "Data Written"


def convert_input(Xin):
    '''
    Convert the QM7 data to the proper format

    This removes all the padding values from the elements and coords
    '''
    new = []
    for z, r in Xin:
        temp = [(NUM_TO_ELE[int(x)], y) for x, y in zip(z, r) if x]
        new.append(zip(*temp))
    return new



if __name__ == "__main__":
    if not os.path.exists("qm7.mat"):
        download_data()
    data = scipy.io.loadmat("qm7.mat")
    P = data['P']

    train_idxs = numpy.ravel(P[1:])
    test_idxs = P[0]

    y = numpy.ravel(data['T'])
    y_train = y[train_idxs]
    y_test = y[test_idxs]

    BOHR_TO_ANGSTROMS = 0.529177249
    R = BOHR_TO_ANGSTROMS * data['R']
    R_train = R[train_idxs, :, :]
    R_test = R[test_idxs, :, :]
    Z_train = data['Z'][train_idxs, :]
    Z_test = data['Z'][test_idxs, :]

    Xin_train = zip(Z_train, R_train)
    Xin_test = zip(Z_test, R_test)

    Xin_train = convert_input(Xin_train)
    Xin_test = convert_input(Xin_test)

    tfs = [
        EncodedBond(n_jobs=-1),
        BagOfBonds(n_jobs=-1),
        CoulombMatrix(n_jobs=-1),
        Connectivity(depth=1, n_jobs=-1),
        Connectivity(depth=2, use_bond_order=True, n_jobs=-1),
    ]
    for tf in tfs:
        print tf
        X_train = tf.fit_transform(Xin_train)
        X_test = tf.transform(Xin_test)

        clf = Ridge()
        clf.fit(X_train, y_train)
        train_error = numpy.abs(clf.predict(X_train) - y_train).mean()
        test_error = numpy.abs(clf.predict(X_test) - y_test).mean()
        print "Train: %.4f Test: %.4f" % (train_error, test_error)
        print 

