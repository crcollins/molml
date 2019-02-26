import os
import unittest

import numpy
from sklearn.pipeline import Pipeline

from molml.fragment import FragmentMap
from molml.molecule import Connectivity

from .constants import METHANE_NUMBERS, MID_NUMBERS
from .constants import DATA_PATH, METHANE_PATH, MID_PATH


ALL = (METHANE_PATH, MID_PATH)
ALL_NUMS = [METHANE_NUMBERS, MID_NUMBERS]
ALL_FEATURES = numpy.array([
    [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
    [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
     [0, 1, 0], [0, 1, 0], [0, 1, 0]]])

LABELS = ['methane', 'mid']


class FragmentMapTest(unittest.TestCase):
    def test_error_if_no_transformer(self):
        with self.assertRaises(ValueError):
            FragmentMap()

    def test_filename_to_label(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans, filename_to_label='identity',
                        label_to_filename=(DATA_PATH, ))
        func = a._get_filename_to_label()
        self.assertEqual(func(METHANE_PATH), METHANE_PATH)

    def test_callable_filename_to_label(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans, filename_to_label=lambda x: x[-5:],
                        label_to_filename=(DATA_PATH, ))
        func = a._get_filename_to_label()
        self.assertEqual(func(METHANE_PATH), METHANE_PATH[-5:])

    def test_invalid_filename_to_label(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans, filename_to_label='bad')
        with self.assertRaises(KeyError):
            a.fit([ALL])

    def test_label_to_filename(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans, label_to_filename=(DATA_PATH, ))
        self.assertEqual(a._get_label_to_filename()('methane'), METHANE_PATH)

    def test_label_to_filename_not_found(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans, label_to_filename=(DATA_PATH, ))
        with self.assertRaises(ValueError):
            a._get_label_to_filename()('not real')

    def test_callable_label_to_filename(self):
        trans = Connectivity(input_type="filename")

        def func(x):
            return os.path.join(DATA_PATH, x)

        a = FragmentMap(transformer=trans, label_to_filename=func)
        self.assertEqual(a._get_label_to_filename()('test'),
                         os.path.join(DATA_PATH, 'test'))

    def test_invalid_label_to_filename(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans, label_to_filename=lambda x: 1)
        with self.assertRaises(KeyError):
            a.fit_transform(ALL)

    def test_bad_input_type(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(input_type='bad', transformer=trans)
        with self.assertRaises(ValueError):
            a.fit([ALL])

    def test_label_input_type(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(input_type='label', transformer=trans,
                        label_to_filename=(DATA_PATH, ))
        a.fit([['methane', 'mid', 'bad']])

    def test_fit(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans)
        a.fit([ALL])
        self.assertEqual(2, a._length)
        expected = {'mid': [2, 3, 4], 'methane': [1, 4, 0]}
        simplified = {x: y.tolist() for x, y in a._x_fragments.items()}
        self.assertEqual(expected, simplified)

    def test_transform_before_fit(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans)
        with self.assertRaises(ValueError):
            a.transform(ALL)

    def test_transform(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans)
        a.fit([ALL])
        res = a.transform([ALL])
        expected = numpy.array([[[1, 4, 0],
                                 [2, 3, 4]]])
        try:
            numpy.testing.assert_array_almost_equal(expected, res)
        except AssertionError as e:
            self.fail(e)

    def test_fit_transform(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans)
        res = a.fit_transform([ALL])
        expected = numpy.array([[[1, 4, 0],
                                 [2, 3, 4]]])
        try:
            numpy.testing.assert_array_almost_equal(expected, res)
        except AssertionError as e:
            self.fail(e)

    def test_get_labels(self):
        trans = Connectivity(input_type="filename")
        a = FragmentMap(transformer=trans)
        res = a.fit_transform([ALL])
        labels = a.get_labels()
        self.assertEqual(res.shape[1] * res.shape[2], len(labels))
        expected = ('0_C', '0_H', '0_O', '1_C', '1_H', '1_O', )
        self.assertEqual(labels, expected)

    def test_get_labels_no_labels(self):
        trans = Pipeline([('Con', Connectivity(input_type="filename"))])
        a = FragmentMap(transformer=trans)
        res = a.fit_transform([ALL])
        labels = a.get_labels()
        self.assertEqual(res.shape[1] * res.shape[2], len(labels))
        expected = ('0_0', '0_1', '0_2', '1_0', '1_1', '1_2', )
        self.assertEqual(labels, expected)


if __name__ == '__main__':
    unittest.main()
