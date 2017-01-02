from __future__ import print_function

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as MAE

from molml.features import EncodedBond, BagOfBonds, Connectivity, CoulombMatrix

from utils import load_qm7


if __name__ == "__main__":
    # This is just boiler plate code to load the data
    Xin_train, Xin_test, y_train, y_test = load_qm7()

    # Change this to make the tranformations parallel
    # Values less than 1 will set to the number of cores the CPU has
    N_JOBS = 1

    # Just a few examples of different features
    tfs = [
        EncodedBond(n_jobs=N_JOBS),
        EncodedBond(spacing="inverse", n_jobs=N_JOBS),
        BagOfBonds(n_jobs=N_JOBS),
        CoulombMatrix(n_jobs=N_JOBS),
        Connectivity(depth=1, n_jobs=N_JOBS),
        Connectivity(depth=2, use_bond_order=True, n_jobs=N_JOBS),
        Connectivity(depth=3, use_coordination=True, n_jobs=N_JOBS),
    ]

    for tf in tfs:
        print(tf)
        X_train = tf.fit_transform(Xin_train)
        X_test = tf.transform(Xin_test)

        # We will not do a hyperparmeter search for simplicity
        clf = Ridge()
        clf.fit(X_train, y_train)
        train_error = MAE(clf.predict(X_train), y_train)
        test_error = MAE(clf.predict(X_test), y_test)
        print("Train MAE: %.4f Test MAE: %.4f" % (train_error, test_error))
        print()
