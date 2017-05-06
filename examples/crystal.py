from __future__ import print_function

import numpy

from molml.features import CoulombMatrix
from molml.crystal import GenerallizedCrystal, EwaldSumMatrix, SineMatrix


# Define some base data
H2_ELES = ['H', 'H']
H2_NUMS = [1, 1]
H2_COORDS = numpy.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
])
H2_CONNS = {
    0: {1: '1'},
    1: {0: '1'},
}
H2_UNIT = numpy.array([
    [2., .5, 0.],
    [.25, 1., 0.],
    [0., .3, 1.],
])

radius = 4.1
input_type = ("elements", "coords", "unit_cell")
X = (H2_ELES, H2_COORDS, H2_UNIT)

if __name__ == "__main__":
    trans = EwaldSumMatrix(input_type=input_type, G_max=3.2, L_max=2.1)
    res = trans.fit_transform([X])
    print(res)

    trans = SineMatrix(input_type=input_type)
    res = trans.fit_transform([X])
    print(res)

    # Example of generallized crystal
    # Any transformer can be used as it just expands the molecule using the
    # unit cell and coordinates.
    cm = CoulombMatrix(input_type=input_type)
    trans = GenerallizedCrystal(transformer=cm,
                                radius=radius)
    res = trans.fit_transform([X])
    print(res)
