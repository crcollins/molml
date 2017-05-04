import unittest
import os

import numpy

from molml.io import read_file_data
from molml.io import read_out_data, read_xyz_data, read_mol2_data
from molml.io import read_cry_data


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ELEMENTS = ['C', 'H', 'H', 'H', 'H']
NUMBERS = [6, 1, 1, 1, 1]
COORDS = [
    [0.99826008, -0.00246000, -0.00436000],
    [2.09021016, -0.00243000, 0.00414000],
    [0.63379005, 1.02686007, 0.00414000],
    [0.62704006, -0.52773003, 0.87811010],
    [0.64136006, -0.50747003, -0.90540005],
]
UNIT = [
    [2.0, 0.5, 0.05],
    [0.0, 2.0, 0.05],
    [0.0, 0.1, 2.0],
]


class IOTest(unittest.TestCase):

    def test_read_file_data(self):
        base_path = os.path.join(DATA_PATH, "methane")
        data = (
                ('.out', read_out_data),
                ('.xyz', read_xyz_data),
                ('.mol2', read_mol2_data),
                ('.cry', read_cry_data),
        )
        for ending, func in data:
            path = base_path + ending
            v1 = func(path)
            v2 = read_file_data(path)
            self.assertEqual(v1.elements.tolist(), v2.elements.tolist())
            self.assertEqual(v1.numbers.tolist(), v2.numbers.tolist())
            self.assertTrue(numpy.allclose(v1.coords, v2.coords))

            self.assertEqual(v1.elements.tolist(), ELEMENTS)
            try:
                numpy.testing.assert_array_almost_equal(
                    v1.coords,
                    COORDS,
                    decimal=3)
            except AssertionError as e:
                self.fail(e)
            self.assertEqual(v1.numbers.tolist(), NUMBERS)

    def test_read_cry_data_unit(self):
        path = os.path.join(DATA_PATH, "methane.cry")
        v = read_cry_data(path)
        try:
            numpy.testing.assert_array_almost_equal(
                v.unit_cell,
                UNIT,
                decimal=3)
        except AssertionError as e:
            self.fail(e)

    def test_read_file_data_error(self):
        path = "garbage.nope"
        with self.assertRaises(ValueError):
            read_file_data(path)


if __name__ == '__main__':
    unittest.main()
