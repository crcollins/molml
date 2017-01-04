import unittest
import os

import numpy

from molml.utils import LazyValues, SMOOTHING_FUNCTIONS
from molml.utils import get_coulomb_matrix, get_element_pairs
from molml.utils import read_file_data, read_out_data, read_xyz_data
from molml.utils import deslugify


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
CONNECTIONS = {
    0: {1: "1", 2: "1", 3: "1", 4: "1"},
    1: {0: "1"},
    2: {0: "1"},
    3: {0: "1"},
    4: {0: "1"},
}


class UtilsTest(unittest.TestCase):

    def test_smoothing_zero_one(self):
        f = SMOOTHING_FUNCTIONS['zero_one']
        values = numpy.array([-1000., -1., -0.5, 0, 0.5, 1., 1000.])
        expected = numpy.array([0., 0., 0., 0., 1., 1., 1.])
        self.assertTrue((f(values) == expected).all())

    def test_smoothing_tanh(self):
        f = SMOOTHING_FUNCTIONS['tanh']
        values = numpy.array([-1000., -1., -0.5, 0, 0.5, 1., 1000.])
        expected = numpy.array([0., 0.11920292, 0.26894142, 0.5,
                                0.73105858, 0.88079708, 1.])
        try:
            numpy.testing.assert_array_almost_equal(
                f(values),
                expected)
        except AssertionError as e:
            self.fail(e)

    def test_smoothing_spike(self):
        f = SMOOTHING_FUNCTIONS['spike']
        values = numpy.array([-1000., -1., -0.5, 0, 0.5, 1., 1000.])
        expected = numpy.array([0., 0., 1., 1., 1., 0., 0.])
        self.assertTrue((f(values) == expected).all())

    def test_get_coulomb_matrix(self):
        res = get_coulomb_matrix([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expected_results = numpy.array([
            [0.5, 1.0],
            [1.0, 0.5]])
        try:
            numpy.testing.assert_array_almost_equal(
                res,
                expected_results)
        except AssertionError as e:
            self.fail(e)

    def test_get_coulomb_matrix_alpha(self):
        nums = [1, 1]
        coords = [[0.0, 0.0, 0.0], [0.0, 0.0, .5]]
        res = get_coulomb_matrix(nums, coords, alpha=2)
        expected_results = numpy.array([
            [0.5, 4.],
            [4., 0.5]])
        try:
            numpy.testing.assert_array_almost_equal(
                res,
                expected_results)
        except AssertionError as e:
            self.fail(e)

    def test_get_coulomb_matrix_use_decay(self):
        nums = [1, 1, 1]
        coords = [[0.0, 0.0, 0.0], [0.0, 0.0, .5], [0.0, 0.5, 0.0]]
        res = get_coulomb_matrix(nums, coords, use_decay=True)
        expected_results = numpy.array([
            [0.5, 1., 1.],
            [1., 0.5, 0.585786],
            [1., 0.585786, 0.5]])
        try:
            numpy.testing.assert_array_almost_equal(
                res,
                expected_results)
        except AssertionError as e:
            self.fail(e)

    def test_get_element_pairs(self):
        res = get_element_pairs(ELEMENTS)
        self.assertEqual(set(res), set([('C', 'H'), ('H', 'H')]))

    def test_read_file_data(self):
        base_path = os.path.join(DATA_PATH, "methane")
        data = (('.out', read_out_data), ('.xyz', read_xyz_data))
        for ending, func in data:
            path = base_path + ending
            e1, n1, c1 = func(path)
            e2, n2, c2 = read_file_data(path)
            self.assertEqual(e1, e2)
            self.assertEqual(n1, n2)
            self.assertTrue((c1 == c2).all())

    def test_read_file_data_error(self):
        path = "garbage.nope"
        with self.assertRaises(ValueError):
            read_file_data(path)

    def test_read_out_data(self):
        path = os.path.join(DATA_PATH, "methane.out")
        elements, numbers, coords = read_out_data(path)
        self.assertEqual(elements, ELEMENTS)
        try:
            numpy.testing.assert_array_almost_equal(
                coords,
                COORDS)
        except AssertionError as e:
            self.fail(e)
        self.assertEqual(numbers, NUMBERS)

    def test_read_xyz_data(self):
        path = os.path.join(DATA_PATH, "methane.xyz")
        elements, numbers, coords = read_xyz_data(path)
        self.assertEqual(elements, ELEMENTS)
        try:
            numpy.testing.assert_array_almost_equal(
                coords,
                COORDS)
        except AssertionError as e:
            self.fail(e)
        self.assertEqual(numbers, NUMBERS)

    def test_deslugify(self):
        string = 'Class__int=1__float=1.__str=string'
        expected = ('Class', {'int': 1, 'float': 1., 'str': 'string'})
        self.assertEqual(deslugify(string), expected)

        string = 'ClassM__none=None__true=True__false=False'
        expected = ('ClassM', {'none': None, 'true': True, 'false': False})
        self.assertEqual(deslugify(string), expected)


class LazyValuesTest(unittest.TestCase):

    def test_all(self):
        a = LazyValues(elements=ELEMENTS, coords=COORDS, numbers=NUMBERS,
                       connections=CONNECTIONS)
        self.assertEqual(a.elements, ELEMENTS)
        self.assertEqual(a.coords, COORDS)
        self.assertEqual(a.numbers, NUMBERS)
        self.assertEqual(a.connections, CONNECTIONS)

    def test_num_from_ele(self):
        a = LazyValues(elements=ELEMENTS)
        self.assertEqual(a.numbers, NUMBERS)

    def test_ele_from_num(self):
        a = LazyValues(numbers=NUMBERS)
        self.assertEqual(a.elements, ELEMENTS)

    def test_no_coords(self):
        a = LazyValues(elements=ELEMENTS, numbers=NUMBERS)
        with self.assertRaises(ValueError):
            a.coords

    def test_no_ele_or_num(self):
        a = LazyValues(coords=COORDS)
        with self.assertRaises(ValueError):
            a.elements
        with self.assertRaises(ValueError):
            a.numbers

    def test_connections(self):
        a = LazyValues(elements=ELEMENTS, coords=COORDS, numbers=NUMBERS)
        self.assertEqual(a.connections, CONNECTIONS)


if __name__ == '__main__':
    unittest.main()
