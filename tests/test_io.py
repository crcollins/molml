import unittest
import os

import numpy

from molml.io import read_file_data
from molml.io import read_out_data, read_xyz_data, read_mol2_data


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


class IOTest(unittest.TestCase):

    def test_read_file_data(self):
        base_path = os.path.join(DATA_PATH, "methane")
        data = (
                ('.out', read_out_data),
                ('.xyz', read_xyz_data),
                ('.mol2', read_mol2_data),
        )
        for ending, func in data:
            path = base_path + ending
            e1, n1, c1 = func(path)
            e2, n2, c2 = read_file_data(path)
            self.assertEqual(e1, e2)
            self.assertEqual(n1, n2)
            self.assertTrue((c1 == c2).all())

            self.assertEqual(e1, ELEMENTS)
            try:
                numpy.testing.assert_array_almost_equal(
                    c1,
                    COORDS,
                    decimal=3)
            except AssertionError as e:
                self.fail(e)
            self.assertEqual(n1, NUMBERS)

    def test_read_file_data_error(self):
        path = "garbage.nope"
        with self.assertRaises(ValueError):
            read_file_data(path)


if __name__ == '__main__':
    unittest.main()
