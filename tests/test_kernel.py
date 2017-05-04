import unittest

import numpy

from molml.kernel import AtomKernel
from molml.atom import Shell

from .constants import METHANE_NUMBERS, MID_NUMBERS
from .constants import METHANE_PATH, MID_PATH


ALL = (METHANE_PATH, MID_PATH)
ALL_NUMS = [METHANE_NUMBERS, MID_NUMBERS]
ALL_FEATURES = numpy.array([
    [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
    [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
     [0, 1, 0], [0, 1, 0], [0, 1, 0]]])
RBF_KERNEL = numpy.array([
    [17., 14.],
    [14., 29.],
])
LAPLACE_KERNEL = numpy.array([
    [17., 2.563417],
    [2.563417, 17.955894],
])
LINEAR_KERNEL = numpy.array([
    [32., 0.],
    [0., 14.],
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
        a = AtomKernel(transformer=trans)
        a.fit(ALL)
        self.assertEqual(ALL_NUMS, [x.tolist() for x in a._numbers])
        self.assertEqual(list(ALL_FEATURES), list(a._features))

    def test_transform_before_fit(self):
        a = AtomKernel()
        with self.assertRaises(ValueError):
            a.transform(ALL)

    def test_transfrom_features(self):
        trans = Shell(input_type="filename")
        feats = trans.fit_transform(ALL)
        a = AtomKernel()
        values = list(zip(feats, ALL_NUMS))
        a.fit(values)
        res = a.transform(values)
        try:
            numpy.testing.assert_array_almost_equal(RBF_KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_transform_transformer(self):
        trans = Shell(input_type="filename")
        a = AtomKernel(transformer=trans)
        a.fit(ALL)
        res = a.transform(ALL)
        try:
            numpy.testing.assert_array_almost_equal(RBF_KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_fit_transform_features(self):
        trans = Shell(input_type="filename")
        feats = trans.fit_transform(ALL)
        a = AtomKernel()
        values = list(zip(feats, ALL_NUMS))
        res = a.fit_transform(values)
        try:
            numpy.testing.assert_array_almost_equal(RBF_KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_fit_transform_transformer(self):
        trans = Shell(input_type="filename")
        a = AtomKernel(transformer=trans)
        res = a.fit_transform(ALL)
        try:
            numpy.testing.assert_array_almost_equal(RBF_KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_same_element(self):
        # Set depth=2 so the comparison is not trivial
        trans = Shell(input_type="filename", depth=2)
        # Set gamma=1 to make the differences more noticeable
        a = AtomKernel(transformer=trans, same_element=False, gamma=1.)
        res = a.fit_transform(ALL)
        expected = numpy.array([[17.00000033, 14.58016505],
                                [14.58016505, 32.76067832]])
        try:
            numpy.testing.assert_array_almost_equal(expected, res)
        except AssertionError as e:
            self.fail(e)

    def test_default_input_type(self):
        a = AtomKernel()
        self.assertEqual("list", a.input_type)

    def test_input_type_mismatch(self):
        trans = Shell(input_type="filename")
        with self.assertRaises(ValueError):
            AtomKernel(input_type="list", transformer=trans)

    def test_input_type_ignored(self):
        a = AtomKernel(input_type="filename")
        self.assertEqual("filename", a.input_type)

    def test_input_type_match(self):
        trans = Shell(input_type="filename")
        AtomKernel(input_type="filename", transformer=trans)

    def test_laplace_kernel(self):
        # Set depth=2 so the comparison is not trivial
        trans = Shell(input_type="filename", depth=2)
        a = AtomKernel(transformer=trans, kernel="laplace", gamma=1.)
        a.fit(ALL)
        res = a.transform(ALL)
        try:
            numpy.testing.assert_array_almost_equal(LAPLACE_KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_custom_kernel(self):
        # Set depth=2 so the comparison is not trivial
        trans = Shell(input_type="filename", depth=2)
        # Simple linear kernel
        a = AtomKernel(transformer=trans,
                       kernel=lambda x, y: numpy.dot(x, numpy.transpose(y)))
        a.fit(ALL)
        res = a.transform(ALL)
        try:
            numpy.testing.assert_array_almost_equal(LINEAR_KERNEL, res)
        except AssertionError as e:
            self.fail(e)

    def test_invalid_kernel(self):
        with self.assertRaises(ValueError):
            trans = Shell(input_type="filename")
            a = AtomKernel(kernel=1, transformer=trans)
            a.fit_transform(ALL)


if __name__ == '__main__':
    unittest.main()
