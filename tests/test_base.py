import os
import unittest

import numpy

from molml.features import Connectivity
from molml.base import BaseFeature, _func_star, MultiFeature
from molml.utils import read_file_data


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

METHANE_PATH = os.path.join(DATA_PATH, "methane.out")
METHANE_ELEMENTS, METHANE_NUMBERS, METHANE_COORDS = read_file_data(
    METHANE_PATH)
METHANE = (METHANE_ELEMENTS, METHANE_COORDS)
METHANE_ATOMS = numpy.array([[1, 4]])


class OtherTest(unittest.TestCase):

    def test__func_star(self):
        res = _func_star((lambda x, y: x + y, 2, 3))
        self.assertEqual(res, 5)


class BaseFeatureTest(unittest.TestCase):

    def test_map_n_jobs_negative(self):
        a = BaseFeature(n_jobs=-1)
        res = a.map(lambda x: x ** 2, range(10))
        self.assertEqual(res, [x ** 2 for x in range(10)])

    def test_map_n_jobs_one(self):
        a = BaseFeature(n_jobs=1)
        res = a.map(lambda x: x ** 2, range(10))
        self.assertEqual(res, [x ** 2 for x in range(10)])

    def test_map_n_jobs_greater(self):
        a = BaseFeature(n_jobs=2)
        res = a.map(lambda x: x ** 2, range(10))
        self.assertEqual(res, [x ** 2 for x in range(10)])

    def test_reduce_n_jobs_negative(self):
        a = BaseFeature(n_jobs=-1)
        res = a.reduce(lambda x, y: x + y, range(10))
        self.assertEqual(res, sum(range(10)))

    def test_reduce_n_jobs_one(self):
        a = BaseFeature(n_jobs=1)
        res = a.reduce(lambda x, y: x + y, range(10))
        self.assertEqual(res, sum(range(10)))

    def test_reduce_n_jobs_greater(self):
        a = BaseFeature(n_jobs=2)
        res = a.reduce(lambda x, y: x + y, range(10))
        self.assertEqual(res, sum(range(10)))

    def test_convert_input_list(self):
        a = BaseFeature(input_type="list")
        data = a.convert_input(METHANE)
        compare_connections = {
            0: {1: "1", 2: "1", 3: "1", 4: "1"},
            1: {0: "1"},
            2: {0: "1"},
            3: {0: "1"},
            4: {0: "1"},
        }
        self.assertEqual(data.connections, compare_connections)
        self.assertEqual((data.elements, data.coords), METHANE)

    def test_convert_input_list_connections(self):
        a = BaseFeature(input_type="list")
        connections = {
            0: {1: "1", 2: "1", 3: "1", 4: "1"},
            1: {0: "1"},
            2: {0: "1"},
            3: {0: "1"},
            4: {0: "1"},
        }
        data = a.convert_input([METHANE[0], METHANE[1], connections])
        self.assertEqual(data.connections, connections)
        self.assertEqual((data.elements, data.coords), METHANE)

    def test_convert_input_filename(self):
        a = BaseFeature(input_type="filename")
        path = os.path.join(os.path.dirname(__file__), "data", "methane.out")
        data = a.convert_input(path)
        self.assertEqual(data.elements, METHANE_ELEMENTS)
        compare_connections = {
            0: {1: "1", 2: "1", 3: "1", 4: "1"},
            1: {0: "1"},
            2: {0: "1"},
            3: {0: "1"},
            4: {0: "1"},
        }
        self.assertEqual(data.connections, compare_connections)
        try:
            numpy.testing.assert_array_almost_equal(
                data.coords, METHANE_COORDS)
        except AssertionError as e:
            self.fail(e)

    def test_convert_input_ele_coords(self):
        a = BaseFeature(input_type=["elements", "coords"])
        data = a.convert_input([METHANE_ELEMENTS, METHANE_COORDS])
        self.assertEqual(data.elements, METHANE_ELEMENTS)
        try:
            numpy.testing.assert_array_almost_equal(
                data.coords, METHANE_COORDS)
        except AssertionError as e:
            self.fail(e)

    def test_convert_input_num_ele(self):
        a = BaseFeature(input_type=["numbers", "elements"])
        data = a.convert_input([METHANE_NUMBERS, METHANE_ELEMENTS])
        self.assertEqual(data.elements, METHANE_ELEMENTS)
        self.assertEqual(data.numbers, METHANE_NUMBERS)

    def test_convert_input_invalid_list(self):
        a = BaseFeature(input_type=["error"])
        with self.assertRaises(TypeError):
            a.convert_input("bad data")

    def test_convert_input_error(self):
        a = BaseFeature(input_type="error")
        with self.assertRaises(ValueError):
            a.convert_input("bad data")

    def test_slugify(self):
        a = Connectivity()
        expected = [
                    'Connectivity',
                    'depth=1',
                    'use_bond_order=False',
                    'use_coordination=False',
                    'add_unknown=False'
                    ]
        self.assertEqual(a.slugify(), '__'.join(expected))


class MultiFeatureTest(unittest.TestCase):

    def test_fit(self):
        feats = [Connectivity(), Connectivity()]
        a = MultiFeature(feats)
        a.fit([METHANE, METHANE])
        self.assertIsNotNone(a.features[0]._base_chains)
        self.assertIsNotNone(a.features[1]._base_chains)

    def test_transform(self):
        feats = [Connectivity(), Connectivity()]
        a = MultiFeature(feats)
        a.fit([METHANE, METHANE])
        res = a.transform([METHANE, METHANE])
        expected = numpy.tile(METHANE_ATOMS, (2, 2))
        self.assertTrue((res == expected).all())

    def test_transform_before_fit(self):
        feats = [Connectivity(), Connectivity()]
        a = MultiFeature(feats)
        with self.assertRaises(ValueError):
            a.transform([METHANE, METHANE])

    def test_fit_transform(self):
        feats = [Connectivity(), Connectivity()]
        a = MultiFeature(feats)
        res = a.fit_transform([METHANE, METHANE])
        expected = numpy.tile(METHANE_ATOMS, (2, 2))
        self.assertTrue((res == expected).all())


if __name__ == '__main__':
    unittest.main()
