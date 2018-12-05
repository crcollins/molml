import os
import unittest
import json
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import numpy

from molml.base import BaseFeature, SetMergeMixin, InputTypeMixin, _func_star
from molml.base import EncodedFeature

from .constants import METHANE_ELEMENTS, METHANE_COORDS, METHANE_PATH
from .constants import METHANE, METHANE_NUMBERS, METHANE_CONNECTIONS


METHANE_ATOMS = numpy.array([[1, 4]])


class TestFeature1(BaseFeature):
    '''
    Some example doc string.

    References
    ----------
    Doe, J. Nature. (2016).
    '''
    LABELS = ('labels', )
    ATTRIBUTES = ('data', )

    def __init__(self, input_type='list', n_jobs=1, data=None, value=None):
        super(TestFeature1, self).__init__(input_type, n_jobs)
        self.labels = ('C', 'B', 'A')
        self.data = data
        self.value = value

    def fit(self, X, y=None):
        self.data = [1]

    def _para_transform(self, X):
        return [1]


class TestFeature2(BaseFeature):
    '''
    Some example doc string.

    References
    ----------
    Doe, J. Nature. (2016).

    Smith, J. Science. (2010).
    '''
    LABELS = ('labels1', 'labels2')
    ATTRIBUTES = ('data1', 'data2')

    def __init__(self, value=None):
        self.labels1 = ('A', 'B', 'C')
        self.labels2 = ('DD', 'CC')
        self.data1 = value
        self.data2 = value


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
    LABELS = None
    ATTRIBUTES = None


class TestFeature4(BaseFeature):
    '''
    Some example doc string.

    References
    ----------
    Doe, J. Nature. (2016).

    Smith, J. Science. (2010).
    '''
    LABELS = (('func', 'labels'), )
    ATTRIBUTES = ('data1', 'data2')

    def __init__(self, value=None):
        self.labels = ('A', 'B', 'C')
        self.data1 = value
        self.data2 = value

    def func(self, labels):
        return labels


class TestFeature5(BaseFeature):
    '''
    '''
    LABELS = (('func', None), )
    ATTRIBUTES = ('data1', 'data2')

    def __init__(self, value=None):
        self.labels = ('A', 'B', 'C')
        self.data1 = value
        self.data2 = value

    def func(self):
        return ['A', 'B', 'C']


#################################################
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
        self.assertEqual(data.connections, METHANE_CONNECTIONS)
        self.assertEqual(data.elements.tolist(), METHANE_ELEMENTS)
        self.assertEqual(data.coords.tolist(), METHANE_COORDS.tolist())

    def test_convert_input_list_numbers(self):
        a = BaseFeature(input_type="list")
        data = a.convert_input([METHANE_NUMBERS, METHANE_COORDS])
        self.assertEqual(data.numbers.tolist(), METHANE_NUMBERS)
        self.assertEqual(data.connections, METHANE_CONNECTIONS)
        self.assertEqual(data.coords.tolist(), METHANE_COORDS.tolist())

    def test_convert_input_list_connections(self):
        a = BaseFeature(input_type="list")
        data = a.convert_input([METHANE_ELEMENTS, METHANE_COORDS,
                                METHANE_CONNECTIONS])
        self.assertEqual(data.connections, METHANE_CONNECTIONS)
        self.assertEqual(data.elements.tolist(), METHANE_ELEMENTS)
        self.assertEqual(data.coords.tolist(), METHANE_COORDS.tolist())

    def test_convert_input_filename(self):
        a = BaseFeature(input_type="filename")
        base_path = os.path.join(os.path.dirname(__file__), "data", "methane")
        for ending in ('.xyz', '.out'):
            path = base_path + ending
            data = a.convert_input(path)
            self.assertEqual(data.elements.tolist(), METHANE_ELEMENTS)
            self.assertEqual(data.connections, METHANE_CONNECTIONS)
            try:
                numpy.testing.assert_array_almost_equal(
                    data.coords, METHANE_COORDS)
            except AssertionError as e:
                self.fail(e)

    def test_convert_input_ele_coords(self):
        a = BaseFeature(input_type=["elements", "coords"])
        data = a.convert_input([METHANE_ELEMENTS, METHANE_COORDS])
        self.assertEqual(data.elements.tolist(), METHANE_ELEMENTS)
        try:
            numpy.testing.assert_array_almost_equal(
                data.coords, METHANE_COORDS)
        except AssertionError as e:
            self.fail(e)

    def test_convert_input_num_ele(self):
        a = BaseFeature(input_type=["numbers", "elements"])
        data = a.convert_input([METHANE_NUMBERS, METHANE_ELEMENTS])
        self.assertEqual(data.elements.tolist(), METHANE_ELEMENTS)
        self.assertEqual(data.numbers.tolist(), METHANE_NUMBERS)

    def test_convert_input_invalid_list(self):
        a = BaseFeature(input_type=["error"])
        with self.assertRaises(TypeError):
            a.convert_input("bad data")

    def test_convert_input_error(self):
        a = BaseFeature(input_type="error")
        with self.assertRaises(ValueError):
            a.convert_input("bad data")

    def test_convert_input_callable(self):
        a = BaseFeature(input_type=lambda x: (x, x ** 2))
        res = a.convert_input(10)
        self.assertEqual(res, (10, 100))

    def test_slugify(self):
        a = TestFeature1()
        expected = [
                    'TestFeature1',
                    'data=None',
                    'value=None',
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
        a = TestFeature1()
        self.assertEqual(a.get_labels(), ('C', 'B', 'A'))
        b = TestFeature2()
        self.assertEqual(b.get_labels(), ('A', 'B', 'C', 'DD', 'CC'))
        c = TestFeature3()
        self.assertEqual(c.get_labels(), tuple())
        d = TestFeature4()
        self.assertEqual(d.get_labels(), ('A', 'B', 'C'))
        e = TestFeature5()
        self.assertEqual(e.get_labels(), ('A', 'B', 'C'))

    def test_check_fit(self):
        a = TestFeature1(data=1)
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
        citation = "MolML https://github.com/crcollins/molml"
        self.assertEqual(citation, BaseFeature.get_citation())
        self.assertEqual("Doe, J. Nature. (2016).",
                         TestFeature1.get_citation())
        expected = "Doe, J. Nature. (2016).\n"
        expected += "Smith, J. Science. (2010)."
        self.assertEqual(expected, TestFeature2.get_citation())
        self.assertEqual(expected, TestFeature3.get_citation())

    def test_save_json(self):
        a = TestFeature1()
        f = StringIO()
        a.save_json(f)
        string = f.getvalue()
        data = json.loads(string)
        base = a.__module__
        expected = {'parameters': {'n_jobs': 1,
                                   'input_type': 'list',
                                   'data': None,
                                   'value': None},
                    'attributes': {'data': None},
                    'transformer': base + '.TestFeature1'}
        self.assertEqual(data, expected)

        path = '/tmp/somefile.json'
        a.save_json(path)
        with open(path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data, expected)

    def test_to_json_no_attributes(self):
        a = TestFeature3()
        data = a.to_json()
        base = a.__module__
        expected = {'parameters': {'n_jobs': 1,
                                   'input_type': 'list'},
                    'attributes': {},
                    'transformer': base + '.TestFeature3'}
        self.assertEqual(data, expected)

    def test_save_json_nested_obj(self):
        a = TestFeature1(value=TestFeature1())
        data = a.to_json()
        base = a.__module__
        expected = {
            'attributes': {'data': None},
            'parameters': {
                'n_jobs': 1,
                'input_type': 'list',
                'value': {
                    'parameters': {
                        'n_jobs': 1,
                        'input_type': 'list',
                        'value': None,
                        'data': None,
                    },
                    'attributes': {'data': None},
                    'transformer': base + '.TestFeature1',
                },
                'data': None,
            },
            'transformer': base + '.TestFeature1'
        }
        self.assertEqual(data, expected)

    def test_transform(self):
        a = TestFeature1()
        a.fit([1])
        res = a.transform([1, 2, 3])
        expected = numpy.array([[1], [1], [1]])
        try:
            numpy.testing.assert_array_almost_equal(res, expected)
        except AssertionError as e:
            self.fail(e)

    def test_fit_transform(self):
        a = TestFeature1()
        res = a.fit_transform([1, 2, 3])
        expected = numpy.array([[1], [1], [1]])
        try:
            numpy.testing.assert_array_almost_equal(res, expected)
        except AssertionError as e:
            self.fail(e)


class TestSetMergeMixin(unittest.TestCase):
    def test_multiple_attributes(self):
        class TestFeature(SetMergeMixin, BaseFeature):
            ATTRIBUTES = ("test1", "test2")

            def __init__(self, *args, **kwargs):
                super(TestFeature, self).__init__(*args, **kwargs)

            def _para_fit(self, X):
                return (set([1, 2, 3]), set([2, 3, 4]))

        a = TestFeature(input_type="filename")
        a.fit([METHANE_PATH, METHANE_PATH])
        self.assertEqual((1, 2, 3), a.test1)
        self.assertEqual((2, 3, 4), a.test2)

    def test_fit(self):
        class TestFeature(SetMergeMixin, BaseFeature):
            ATTRIBUTES = ("test1", )

            def __init__(self, *args, **kwargs):
                super(TestFeature, self).__init__(*args, **kwargs)

            def _para_fit(self, X):
                return set([1, 2, 3])

        a = TestFeature(input_type="filename")
        a.fit([METHANE_PATH, METHANE_PATH])
        self.assertEqual((1, 2, 3), a.test1)


class Feature(InputTypeMixin, BaseFeature):

    def __init__(self, input_type=None, transformer=None, *args, **kwargs):
        super(Feature, self).__init__(input_type=input_type, *args, **kwargs)
        self.check_transformer(transformer)
        self.transformer = transformer


class TestInputTypeMixin(unittest.TestCase):

    def test_input_type_default(self):
        a = Feature()
        self.assertEqual("list", a.input_type)

    def test_input_type_mismatch(self):
        trans = BaseFeature(input_type="filename")
        with self.assertRaises(ValueError):
            Feature(input_type="list", transformer=trans)

    def test_input_type_match(self):
        trans = BaseFeature(input_type="filename")
        a = Feature(input_type="filename", transformer=trans)
        self.assertEqual("filename", a.input_type)

    def test_input_type_normal(self):
        a = Feature(input_type="filename")
        self.assertEqual("filename", a.input_type)

    def test_input_type_from_param(self):
        trans = BaseFeature(input_type="filename")
        a = Feature(transformer=trans)
        self.assertEqual("filename", a.input_type)


class TestEncodedFeature(unittest.TestCase):

    def test_encode_values(self):
        a = EncodedFeature(segments=5)
        data = [((0, ), 3, 1), (None, 1, 1), ((1, ), 3, 2)]
        res = a.encode_values(data, (2, ))
        expected = [
            0, 1.997889e-159, 5.399097e-2, 8.363952e-210, 0,
            0, 3.995779e-159, 1.079819e-1, 1.672790e-209, 0]
        try:
            numpy.testing.assert_array_almost_equal(
                res,
                expected)
        except AssertionError as e:
            self.fail(e)

    def test_encode_values_saved_length(self):
        a = EncodedFeature(segments=5)
        data = [((0, 1), 3, 1), ((1, 0), 3, 2), (None, 1, 1)]
        res = a.encode_values(data, (2, 2), saved_lengths=1)

        expected = numpy.zeros((2, 10))
        expected = numpy.array([
            [0, 0, 0, 0, 0, 0, 1.997889e-159, 5.399097e-002, 8.363952e-210, 0],
            [0, 3.995779e-159, 1.079819e-1, 1.672790e-209, 0, 0, 0, 0, 0, 0],
        ])
        try:
            numpy.testing.assert_array_almost_equal(
                res,
                expected)
        except AssertionError as e:
            self.fail(e)

    def test_get_encoded_labels(self):
        a = EncodedFeature(segments=3, start=1., end=3.)
        labels = a.get_encoded_labels([('A', 'B'), ('C', 'D')])
        expected = [
            'A-B_1.0', 'A-B_2.0', 'A-B_3.0',
            'C-D_1.0', 'C-D_2.0', 'C-D_3.0',
        ]
        self.assertEqual(labels, expected)


if __name__ == '__main__':
    unittest.main()
