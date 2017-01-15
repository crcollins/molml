from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.pipeline import FeatureUnion

from molml.features import EncodedBond, Connectivity

from utils import load_qm7


if __name__ == "__main__":
    # This is just boiler plate code to load the data
    Xin_train, Xin_test, y_train, y_test = load_qm7()

    feats = [
        ("encoded_bond", EncodedBond(n_jobs=-1, max_depth=3)),
        ("atom_count", Connectivity(depth=1, n_jobs=-1)),
        ("angle_count", Connectivity(depth=3, use_coordination=True,
                                     n_jobs=-1)),
    ]

    full_feat = FeatureUnion(feats)
    X_train = full_feat.fit_transform(Xin_train)
    X_test = full_feat.transform(Xin_test)

    clfs = [
        Ridge(alpha=0.01),
        KernelRidge(alpha=1e-9, gamma=1e-5, kernel="rbf"),
    ]
    for clf in clfs:
        print(clf)
        clf.fit(X_train, y_train)
        train_error = MAE(clf.predict(X_train), y_train)
        test_error = MAE(clf.predict(X_test), y_test)
        print("Train MAE: %.4f Test MAE: %.4f" % (train_error, test_error))
        print()
