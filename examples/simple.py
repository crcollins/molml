from __future__ import print_function

from molml.features import CoulombMatrix
from molml.features import LocalCoulombMatrix
from molml.kernel import AtomKernel

from molml.utils import LazyValues

# Define some base data
H2_ELES = ['H', 'H']
H2_NUMS = [1, 1]
H2_COORDS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
]
H2_CONNS = {
    0: {1: '1'},
    1: {0: '1'},
}

HCN_ELES = ['H', 'C', 'N']
HCN_NUMS = [1, 6, 7]
HCN_COORDS = [
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
]
HCN_CONNS = {
    0: {1: '1'},
    1: {0: '1', 2: '3'},
    2: {1: '3'},
}


if __name__ == "__main__":
    # Example of Generating the Coulomb matrix with just elements and coords.
    feat = CoulombMatrix()
    H2 = (H2_ELES, H2_COORDS)
    HCN = (HCN_ELES, HCN_COORDS)
    feat.fit([H2, HCN])
    print("Transformed H2")
    print(feat.transform([H2]))
    print("H2 and HCN transformed")
    print(feat.transform([H2, HCN]))
    print()

    # Example of generating the Coulomb matrix with elements, coords, and
    # connections.
    feat = CoulombMatrix()
    H2_conn = (H2_ELES, H2_COORDS, H2_CONNS)
    HCN_conn = (HCN_ELES, HCN_COORDS, HCN_CONNS)
    print(feat.fit_transform([H2_conn, HCN_conn]))
    print()

    # Example of generating the Coulomb matrix using a specified input_type
    print("User specified input_type")
    feat = CoulombMatrix(input_type=("coords", "numbers"))
    H2_spec = (H2_COORDS, H2_NUMS)
    HCN_spec = (HCN_COORDS, HCN_NUMS)
    print(feat.fit_transform([H2_spec, HCN_spec]))
    print()

    # Example of generating the Local Coulomb matrix (atom-wise
    # representation)
    print("Atom feature")
    feat = LocalCoulombMatrix()
    print(feat.fit_transform([H2, HCN]))

    # Example of generating AtomKernel
    print("Atom Kernel")
    feat = AtomKernel(transformer=LocalCoulombMatrix())
    print(feat.fit_transform([H2, HCN]))

    # Example of using arbitrary function to load data
    # This example is useless, but it shows the possibility
    feat = CoulombMatrix(input_type=lambda x: LazyValues(elements=HCN_ELES,
                                                         coords=HCN_COORDS))
    feat.fit_transform(list(range(10)))
