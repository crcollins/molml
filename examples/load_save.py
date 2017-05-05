from __future__ import print_function

import numpy

from molml.features import CoulombMatrix
from molml.crystal import GenerallizedCrystal
from molml.utils import load_json


# Define some base data
H2_ELES = ['H', 'H']
H2_COORDS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
]
H2_UNIT = numpy.array([
    [2., .5, 0.],
    [.25, 1., 0.],
    [0., .3, 1.],
])
H2 = (H2_ELES, H2_COORDS)
H2_FULL = (H2_ELES, H2_COORDS, H2_UNIT)

HCN_ELES = ['H', 'C', 'N']
HCN_COORDS = [
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
]
HCN = (HCN_ELES, HCN_COORDS)


if __name__ == "__main__":
    # Example of fitting the Coulomb matrix and then saving it
    feat = CoulombMatrix()
    feat.fit([H2, HCN])
    print("Saving Model")
    feat.save_json("coulomb_model.json")

    print("Loading Model")
    feat2 = load_json("coulomb_model.json")
    print(feat2.transform([H2, HCN]))

    # Example of fitting a generallized crystal with the Coulomb matrix and
    # then saving it
    input_type = ("elements", "coords", "unit_cell")
    radius = 4.1
    feat = CoulombMatrix(input_type=input_type)
    crystal = GenerallizedCrystal(transformer=feat, radius=radius)
    feat.fit([H2_FULL])
    print("Saving Model")
    feat.save_json("coulomb_crystal_model.json")

    print("Loading Model")
    feat2 = load_json("coulomb_crystal_model.json")
    print(feat2.transform([H2_FULL]))
