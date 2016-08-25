import os
import unittest

import numpy

from molml.atom import Shell, LocalCoulombMatrix
from molml.utils import read_file_data


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

METHANE_PATH = os.path.join(DATA_PATH, "methane.out")
METHANE_ELEMENTS, METHANE_NUMBERS, METHANE_COORDS = read_file_data(
    METHANE_PATH)
METHANE = (METHANE_ELEMENTS, METHANE_COORDS)
METHANE2 = (METHANE[0], 2 * METHANE[1])

BIG_PATH = os.path.join(DATA_PATH, "big.out")
BIG_ELEMENTS, BIG_NUMBERS, BIG_COORDS = read_file_data(BIG_PATH)
BIG = (BIG_ELEMENTS, BIG_COORDS)

MID_PATH = os.path.join(DATA_PATH, "mid.out")
MID_ELEMENTS, MID_NUMBERS, MID_COORDS = read_file_data(MID_PATH)
MID = (MID_ELEMENTS, MID_COORDS)

ALL_DATA = [METHANE, MID, BIG]

BASE_SHELL = numpy.array([
    [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [1, 0, 0, 0]],
    [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
     [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [1, 0, 0, 0]],
    [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0],
     [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
     [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
     [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1],
     [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [1, 0, 0, 0]]
])
BASE_LOCAL_COULOMB = numpy.array([
    numpy.array([
        [36.85810519942594, 3.5389458702468084, 3.5389458702468084, 0.5],
        [0.5, 3.5387965701194934, 3.5387965701194974, 36.85810519942594],
        [0.5, 3.5389458702468084, 3.5389458702468084, 36.85810519942594],
        [0.5, 3.538447971815893, 3.538447971815893, 36.85810519942594],
        [0.5, 3.52229970767669, 3.52229970767669, 36.85810519942594]
    ]),
    numpy.array([
        [36.85810519942594, 4.642192257970912,
            4.642192257970912, 36.85810519942594],
        [36.85810519942594, 6.483079598556282,
            6.483079598556282, 73.51669471981023],
        [73.51669471981023, 8.650089711338763,
            8.650089711338763, 73.51669471981023],
        [73.51669471981023, 8.650089711338763,
            8.650089711338763, 73.51669471981023],
        [73.51669471981023, 10.698256448660478,
            10.698256448660478, 73.51669471981023],
        [73.51669471981023, 10.698256448660478,
            10.698256448660478, 73.51669471981023],
        [0.5, 0.044832923800298255, 0.044832923800298255, 0.5],
        [0.5, 1.0, 1.0, 0.5],
        [0.5, 1.0, 1.0, 0.5]
    ]),
])


class ShellTest(unittest.TestCase):

    def test_fit(self):
        a = Shell(depth=1)
        a.fit(ALL_DATA)
        self.assertEqual(a._elements,
                         set(['N', 'C', 'O', 'H']))

    def test_fit_use_coordination(self):
        a = Shell(depth=1, use_coordination=True)
        a.fit(ALL_DATA)
        self.assertEqual(a._elements,
                         set(['H0', 'H1', 'O2', 'C4', 'N1', 'C3', 'C2', 'N2',
                              'N3', 'C1', 'O1', 'O0']))

    def test_transform(self):
        a = Shell()
        a.fit(ALL_DATA)
        self.assertTrue((a.transform(ALL_DATA) == BASE_SHELL).all())

    def test_transform_use_coordination(self):
        a = Shell(depth=1, use_coordination=True)
        a.fit([MID])
        expected_results = numpy.array([
            [[0, 0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0]]
        ])
        self.assertTrue((a.transform([MID]) == expected_results).all())

    def test_transform_depth2(self):
        a = Shell(depth=2)
        a.fit(ALL_DATA)
        expected_results = numpy.array([
            [[4, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
             [0, 1, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0],
             [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0],
             [1, 0, 0, 0]],
            [[0, 1, 1, 1], [1, 2, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0],
             [0, 2, 0, 1], [0, 2, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
             [1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
             [0, 3, 0, 0], [1, 2, 0, 0], [0, 2, 1, 0], [0, 2, 0, 1],
             [0, 2, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0],
             [0, 2, 0, 0], [3, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0],
             [0, 1, 0, 0], [0, 2, 1, 0], [0, 3, 0, 0], [0, 3, 0, 0],
             [0, 2, 1, 0], [0, 2, 0, 0], [0, 2, 0, 0], [1, 1, 0, 0],
             [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 2],
             [0, 3, 0, 0], [1, 2, 0, 0], [0, 3, 0, 0], [1, 1, 0, 1],
             [0, 2, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [2, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 0, 1], [0, 2, 0, 0], [1, 1, 0, 0],
             [0, 1, 0, 0]]
        ])
        self.assertTrue((a.transform(ALL_DATA) == expected_results).all())

    def test_transform_depth3(self):
        # This is to test loop backs in the breadth-first search
        a = Shell(depth=3)
        a.fit(ALL_DATA)
        expected_results = numpy.array([
            [[0, 0, 0, 0], [3, 0, 0, 0], [3, 0, 0, 0], [3, 0, 0, 0],
             [3, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0],
             [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[2, 2, 0, 0], [1, 1, 1, 1], [2, 2, 0, 0], [1, 2, 0, 1],
             [1, 4, 0, 0], [0, 3, 1, 0], [0, 2, 0, 0], [0, 2, 0, 0],
             [0, 1, 0, 1], [0, 1, 0, 0], [0, 2, 0, 0], [1, 3, 0, 0],
             [1, 2, 1, 2], [0, 3, 1, 0], [1, 3, 0, 1], [0, 3, 2, 0],
             [0, 3, 1, 0], [0, 2, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0],
             [3, 2, 0, 0], [0, 1, 0, 0], [2, 0, 1, 0], [2, 0, 1, 0],
             [2, 0, 1, 0], [0, 4, 0, 1], [0, 4, 1, 0], [0, 3, 1, 1],
             [0, 5, 0, 0], [0, 4, 0, 0], [1, 2, 0, 0], [0, 1, 0, 0],
             [0, 1, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [2, 3, 0, 0],
             [1, 2, 1, 2], [0, 4, 0, 0], [2, 2, 0, 1], [0, 3, 0, 0],
             [1, 2, 0, 1], [0, 2, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1],
             [1, 1, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0], [0, 1, 0, 0],
             [0, 1, 0, 0]]
        ])
        self.assertTrue((a.transform(ALL_DATA) == expected_results).all())

    def test_small_to_large_transform(self):
        a = Shell()
        a.fit([METHANE])
        expected_results = numpy.array([
            [[0, 1], [1, 0], [1, 0], [1, 0], [1, 0]],
            [[0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0],
             [1, 0]],
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [1, 0], [1, 0],
             [0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
             [0, 0], [1, 0], [0, 0], [1, 0], [0, 0], [0, 1], [1, 0], [1, 0],
             [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 1], [0, 1],
             [1, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
             [0, 0], [1, 0], [1, 0], [0, 0], [1, 0], [1, 0], [0, 1], [0, 1],
             [1, 0]]
        ])
        self.assertTrue((a.transform(ALL_DATA) == expected_results).all())

    def test_large_to_small_transform(self):
        a = Shell()
        a.fit([BIG])
        self.assertTrue((a.transform(ALL_DATA) == BASE_SHELL).all())

    def test_transform_before_fit(self):
        a = Shell()
        with self.assertRaises(ValueError):
            a.transform(ALL_DATA)

    def test_fit_transform(self):
        a = Shell()
        self.assertTrue((a.fit_transform(ALL_DATA) == BASE_SHELL).all())


class LocalCoulombMatrixTest(unittest.TestCase):

    def test_fit(self):
        a = LocalCoulombMatrix()
        # This should not do anything
        a.fit(ALL_DATA)

    def test_transform(self):
        a = LocalCoulombMatrix(max_occupancy=1)
        a.fit([METHANE, MID])
        m = a.transform([METHANE, MID])
        try:
            mm = [numpy.linalg.norm(x) for x in (BASE_LOCAL_COULOMB - m)]
            numpy.testing.assert_array_almost_equal(
                mm,
                [0.0, 0.0])
        except AssertionError as e:
            self.fail(e)

    def test_transform_reduced(self):
        a = LocalCoulombMatrix(max_occupancy=1, use_reduced=True)
        a.fit([METHANE, MID])
        m = a.transform([METHANE, MID])
        expected_results = numpy.array([
            [
                [37.3581052, 4.03894587],
                [37.3581052, 40.39690177],
                [37.3581052, 40.39705107],
                [37.3581052, 40.39655317],
                [37.3581052, 40.38040491]
            ],
            [
                [73.7162104, 41.50029746],
                [110.37479992, 79.99977432],
                [147.03338944, 82.16678443],
                [147.03338944, 82.16678443],
                [147.03338944, 84.21495117],
                [147.03338944, 84.21495117],
                [1., 0.54483292],
                [1., 1.5],
                [1., 1.5]
            ]])

        try:
            mm = [numpy.linalg.norm(x) for x in (expected_results - m)]
            numpy.testing.assert_array_almost_equal(
                mm,
                [0.0, 0.0])
        except AssertionError as e:
            self.fail(e)

    def test_transform_alpha(self):
        a = LocalCoulombMatrix(max_occupancy=1, alpha=2., use_reduced=True)
        a.fit([METHANE, MID])
        m = a.transform([METHANE, MID])
        expected_results = numpy.array([
            [
                [37.3581052, 5.53182436],
                [37.3581052, 41.8898588],
                [37.3581052, 41.88992956],
                [37.3581052, 41.88969357],
                [37.3581052, 41.88202775]
            ],
            [
                [73.7162104, 55.04573231],
                [110.37479992, 98.1442485],
                [147.03338944, 106.36100805],
                [147.03338944, 106.36100805],
                [147.03338944, 108.77198683],
                [147.03338944, 108.77198683],
                [1., 0.85524858],
                [1., 1.5],
                [1., 1.5]
            ]])
        try:
            mm = [numpy.linalg.norm(x) for x in (expected_results - m)]
            numpy.testing.assert_array_almost_equal(
                mm,
                [0.0, 0.0])
        except AssertionError as e:
            self.fail(e)

    def test_transform_max_occupancy(self):
        a = LocalCoulombMatrix(max_occupancy=5)
        a.fit([METHANE, MID])
        m = a.transform([METHANE, MID])
        # Reduce to a sum to save space
        expected_results = [337.53938456166259, 3019.413939202841]

        try:
            numpy.testing.assert_array_almost_equal(
                [x.sum() for x in m],
                expected_results)
        except AssertionError as e:
            self.fail(e)

    def test_transform_before_fit(self):
        a = LocalCoulombMatrix(max_occupancy=1)
        # This should not raise an error, becaues no fitting is needed
        a.transform(ALL_DATA)

    def test_fit_transform(self):
        a = LocalCoulombMatrix(max_occupancy=1)
        m = a.fit_transform([METHANE, MID])
        try:
            mm = [numpy.linalg.norm(x) for x in (BASE_LOCAL_COULOMB - m)]
            numpy.testing.assert_array_almost_equal(
                mm,
                [0.0, 0.0])
        except AssertionError as e:
            self.fail(e)

if __name__ == '__main__':
    unittest.main()
