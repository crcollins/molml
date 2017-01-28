import os
import unittest

import numpy

from molml.kernel import AtomKernel
from molml.atom import Shell
from molml.utils import read_file_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
METHANE_PATH = os.path.join(DATA_PATH, "methane.out")
MID_PATH = os.path.join(DATA_PATH, "mid.out")

_, METHANE_NUMS, _ = read_file_data(METHANE_PATH)
_, MID_NUMS, _ = read_file_data(MID_PATH)
ALL = (METHANE_PATH, MID_PATH)
ALL_NUMS = [METHANE_NUMS, MID_NUMS]
ALL_FEATURES = numpy.array([
    [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
    [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
     [0, 1, 0], [0, 1, 0], [0, 1, 0]]])
KERNEL = numpy.array([
    [17., 14.],
    [14., 29.],
])


class AtomKernelTest(unittest.TestCase):
    def test_fit_features(self):
        trans = Shell(input_type="filename")
        feats = trans.fit_transform(ALL)
        a = AtomKernel()
        values = list(zip(feats, ALL_NUMS))
        a.fit(values)
        self.assertEqual(ALL_NUMS, list(a._numbers))
        self.assertEqual(list(ALL_FEATURES), list(a._features))

    def test_fit_transformer(self):
        trans = Shell(input_type="filename")
        a = AtomKernel(input_type="filename", transformer=trans)
        a.fit(ALL)
        self.assertEqual(ALL_NUMS, list(a._numbers))
        self.assertEqual(list(ALL_FEATURES), list(a._features))

    def test_transfrom_features(self):
        trans = Shell(input_type="filename")
        feats = trans.fit_transform(ALL)
        a = AtomKernel()
        values = list(zip(feats, ALL_NUMS))
        a.fit(values)
        res = a.transform(values)
        try:
            numpy.testing.assert_array_almost_equal(KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_transform_transformer(self):
        trans = Shell(input_type="filename")
        a = AtomKernel(input_type="filename", transformer=trans)
        a.fit(ALL)
        res = a.transform(ALL)
        try:
            numpy.testing.assert_array_almost_equal(KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_fit_transform_features(self):
        trans = Shell(input_type="filename")
        feats = trans.fit_transform(ALL)
        a = AtomKernel()
        values = list(zip(feats, ALL_NUMS))
        res = a.fit_transform(values)
        try:
            numpy.testing.assert_array_almost_equal(KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_fit_transform_transformer(self):
        trans = Shell(input_type="filename")
        a = AtomKernel(input_type="filename", transformer=trans)
        res = a.fit_transform(ALL)
        try:
            numpy.testing.assert_array_almost_equal(KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_same_element(self):
        # Set depth=2 so the comparison is not trivial
        trans = Shell(input_type="filename", depth=2)
        # Set gamma=1 to make the differences more noticeable
        a = AtomKernel(input_type="filename", transformer=trans,
                       same_element=False, gamma=1.)
        res = a.fit_transform(ALL)
        expected = numpy.array([[17.00000033, 14.58016505],
                                [14.58016505, 32.76067832]])
        try:
            numpy.testing.assert_array_almost_equal(expected, res)
        except AssertionError as e:
            self.fail(e)


if __name__ == '__main__':
    unittest.main()
