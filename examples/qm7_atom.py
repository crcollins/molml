import numpy
from scipy.spatial.distance import cdist

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE

from molml.features import LocalEncodedBond
from molml.utils import ELE_TO_NUM

from utils import load_qm7


def compute_kernel(a_feats, b_feats, a_nums, b_nums, gamma, symmetric=True):
    '''
    Compute a Gaussian kernel between molecules based on atom features

    Parameters
    ----------
        a_feats : list of numpy.array, shape=(n_molecules_a, )
            Each array is of shape (n_atoms, n_features), where n_atoms is
            for that particular molecule.

        b_feats : list of numpy.array, shape=(n_molecules_b, )
            Each array is of shape (n_atoms, n_features), where n_atoms is
            for that particular molecule.

        a_nums : list of lists, shape=(n_molecules_a, )
            Contains all the atom elements for each molecule in group a

        b_nums : list of lists, shape=(n_molecules_b, )
            Contains all the atom elements for each molecule in group b

        gamma : float
            The hyperparameter to use for the width of the gaussian kernel

        symmetric : bool, default=True
            Whether or not the kernel is symmetric. This is just to cut the
            computational cost in half.

    Returns
    -------
        kernel : numpy.array, shape=(n_molecules_a, n_molecules_b)
            The kernel matrix between the two sets of molecules
    '''

    kernel = numpy.zeros((len(a_feats), len(b_feats)))
    for i, (x, x_nums) in enumerate(zip(a_feats, a_nums)):
        for j, (y, y_nums) in enumerate(zip(b_feats, b_nums)):
            if symmetric and j > i:
                continue

            # Mask to make sure only elements of the same type are compared
            mask = numpy.equal.outer(x_nums, y_nums)
            block = cdist(x, y, 'sqeuclidean')
            block *= -gamma
            numpy.exp(block, block)
            kernel[i, j] = (block * mask).sum()

            if symmetric:
                kernel[j, i] = kernel[i, j]
    return kernel


if __name__ == "__main__":
    # This is just boiler plate code to load the data
    Xin_train, Xin_test, y_train, y_test = load_qm7()

    # Look at just a few examples to be quick
    n_train = 100
    n_test = 10
    Xin_train = Xin_train[:n_train]
    y_train = y_train[:n_train]
    Xin_test = Xin_test[:n_test]
    y_test = y_test[:n_test]

    elements_train, _ = zip(*Xin_train)
    elements_test, _ = zip(*Xin_test)
    # Convert the elements to numbers
    numbers_train = [[ELE_TO_NUM[x] for x in mol] for mol in elements_train]
    numbers_test = [[ELE_TO_NUM[x] for x in mol] for mol in elements_test]

    feat = LocalEncodedBond(n_jobs=-1)
    X_train = feat.fit_transform(Xin_train)
    X_test = feat.transform(Xin_test)

    gamma = 1e-7
    alpha = 1e-7
    K_train = compute_kernel(X_train, X_train,
                             numbers_train, numbers_train,
                             gamma)
    K_test = compute_kernel(X_test, X_train,
                            numbers_test, numbers_train,
                            gamma, symmetric=False)
    clf = KernelRidge(alpha=alpha, kernel="precomputed")
    clf.fit(K_train, y_train)
    train_error = MAE(clf.predict(K_train), y_train)
    test_error = MAE(clf.predict(K_test), y_test)
    print("Train MAE: %.4f Test MAE: %.4f" % (train_error, test_error))
    print()
