from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE

from molml.features import LocalEncodedBond
from molml.kernel import AtomKernel

from utils import load_qm7


if __name__ == "__main__":
    # This is just boiler plate code to load the data
    Xin_train, Xin_test, y_train, y_test = load_qm7()

    # Look at just a few examples to be quick
    n_train = 200
    n_test = 200
    Xin_train = Xin_train[:n_train]
    y_train = y_train[:n_train]
    Xin_test = Xin_test[:n_test]
    y_test = y_test[:n_test]

    gamma = 1e-7
    alpha = 1e-7
    kern = AtomKernel(gamma=gamma, transformer=LocalEncodedBond(n_jobs=-1),
                      n_jobs=-1)
    K_train = kern.fit_transform(Xin_train)
    K_test = kern.transform(Xin_test)

    clf = KernelRidge(alpha=alpha, kernel="precomputed")
    clf.fit(K_train, y_train)
    train_error = MAE(clf.predict(K_train), y_train)
    test_error = MAE(clf.predict(K_test), y_test)
    print("Train MAE: %.4f Test MAE: %.4f" % (train_error, test_error))
    print()
