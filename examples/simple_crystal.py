from __future__ import print_function

import numpy

from molml.utils import LazyValues
from molml.io import read_cry_data

from utils import plot_cell


# Define some base data
H_ELES = ['H']
H_NUMS = [1]
H_COORDS = numpy.array([
    [0.0, 0.0, 0.0],
])
H_CONNS = {
    0: {},
}
H_UNIT = numpy.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
])
radius = 3


if __name__ == "__main__":
    vals = LazyValues(coords=H_COORDS, unit_cell=H_UNIT,
                      elements=H_ELES, numbers=H_NUMS)
    vals.fill_in_crystal(radius=radius)
    plot_cell(vals.coords, radius, vals.unit_cell,
              connections=vals.connections)

    vals = read_cry_data("../tests/data/methane.cry")
    vals.fill_in_crystal(radius=radius)
    plot_cell(vals.coords, radius, vals.unit_cell,
              connections=vals.connections)
