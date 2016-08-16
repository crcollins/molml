import numpy
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE

from molml.features import EncodedBond, Connectivity

from utils import load_qm7


if __name__ == "__main__":
    # This is just boiler plate code to load the data
    Xin_train, Xin_test, y_train, y_test = load_qm7()

    feats = [
        EncodedBond(n_jobs=-1, max_depth=3),
        Connectivity(depth=1, n_jobs=-1),
        Connectivity(depth=3, use_coordination=True, n_jobs=-1),
    ]
    train_feats = []
    test_feats = []
    for tf in feats:
        X_train = tf.fit_transform(Xin_train)
        X_test = tf.transform(Xin_test)
        train_feats.append(X_train)
        test_feats.append(X_test)

    X_train = numpy.hstack(train_feats)
    X_test = numpy.hstack(test_feats)

    clfs = [
        Ridge(alpha=0.01),
        KernelRidge(alpha=1e-9, gamma=1e-5, kernel="rbf"),
    ]
    for clf in clfs:
        print clf
        clf.fit(X_train, y_train)
        train_error = MAE(clf.predict(X_train), y_train)
        test_error = MAE(clf.predict(X_test), y_test)
        print "Train MAE: %.4f Test MAE: %.4f" % (train_error, test_error)
        print
