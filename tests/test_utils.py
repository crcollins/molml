import unittest

import numpy

from molml.utils import LazyValues, SMOOTHING_FUNCTIONS


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
