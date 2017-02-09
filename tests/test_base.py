import os
import unittest

import numpy

from molml.features import Connectivity
from molml.base import BaseFeature, SetMergeMixin, _func_star
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
        base_path = os.path.join(os.path.dirname(__file__), "data", "methane")
        for ending in ('.xyz', '.out'):
            path = base_path + ending
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

    def test_get_params(self):
        a = BaseFeature(n_jobs=10)
        expected = {"input_type": "list", "n_jobs": 10}
        self.assertEqual(a.get_params(), expected)

    def test_set_params(self):
        a = BaseFeature(n_jobs=10)
        new = {
                "input_type": "coords",
                "n_jobs": 100,
                "fake": None,
        }
        a.set_params(**new)
        self.assertEqual(a.input_type, "coords")
        self.assertEqual(a.n_jobs, 100)
        with self.assertRaises(AttributeError):
            a.fake

    def test_get_labels(self):
        class TestFeature1(BaseFeature):
            LABELS = ('labels', )

            def __init__(self):
                self.labels = ('C', 'B', 'A')

        class TestFeature2(BaseFeature):
            LABELS = ('labels1', 'labels2')

            def __init__(self):
                self.labels1 = ('A', 'B', 'C')
                self.labels2 = ('DD', 'CC')

        class TestFeature3(BaseFeature):
            LABELS = None

        a = TestFeature1()
        self.assertEqual(a.get_labels(), ('A', 'B', 'C'))
        b = TestFeature2()
        self.assertEqual(b.get_labels(), ('A', 'B', 'C', 'CC', 'DD'))
        c = TestFeature3()
        self.assertEqual(c.get_labels(), tuple())

    def test_check_fit(self):
        class TestFeature1(BaseFeature):
            ATTRIBUTES = ('data', )

            def __init__(self, value=None):
                self.data = value

        class TestFeature2(BaseFeature):
            ATTRIBUTES = ('data1', 'data2')

            def __init__(self, value=None):
                self.data1 = value
                self.data2 = value

        class TestFeature3(BaseFeature):
            ATTRIBUTES = None

        a = TestFeature1(value=1)
        self.assertIsNone(a.check_fit())
        b = TestFeature2(value=1)
        self.assertIsNone(b.check_fit())
        c = TestFeature3()
        self.assertIsNone(c.check_fit())

        with self.assertRaises(ValueError):
            a = TestFeature1()
            a.check_fit()

        with self.assertRaises(ValueError):
            b = TestFeature2()
            b.check_fit()

    def test_get_citation(self):
        class TestFeature(BaseFeature):
            '''
            Some example doc string.

            References
            ----------
            Doe, J. Nature. (2016).
            '''

        class TestFeature2(BaseFeature):
            '''
            Some example doc string.

            References
            ----------
            Doe, J. Nature. (2016).

            Smith, J. Science. (2010).
            '''

        class TestFeature3(BaseFeature):
            '''
            Some example doc string.

            References
            ----------
            Doe, J. Nature. (2016).

            Smith, J. Science. (2010).

            Other
            -----
            Something else.
            '''

        citation = "MolML https://github.com/crcollins/molml"
        self.assertEqual(citation, BaseFeature.get_citation())
        self.assertEqual("Doe, J. Nature. (2016).",
                         TestFeature.get_citation())
        expected = "Doe, J. Nature. (2016).\n"
        expected += "Smith, J. Science. (2010)."
        self.assertEqual(expected, TestFeature2.get_citation())
        self.assertEqual(expected, TestFeature3.get_citation())


class TestSetMergeMixin(unittest.TestCase):
    def test_multiple_attributes(self):
        class TestFeature(SetMergeMixin, BaseFeature):
            ATTRIBUTES = ("test1", "test2")

        a = TestFeature()
        with self.assertRaises(NotImplementedError):
            a.fit([])

    def test_fit(self):
        class TestFeature(SetMergeMixin, BaseFeature):
            ATTRIBUTES = ("test1", )

            def __init__(self, *args, **kwargs):
                super(TestFeature, self).__init__(*args, **kwargs)

            def _para_fit(self, X):
                return set([1, 2, 3])

        a = TestFeature(input_type="filename")
        a.fit([METHANE_PATH, METHANE_PATH])
        self.assertEqual({1, 2, 3}, a.test1)


if __name__ == '__main__':
    unittest.main()
