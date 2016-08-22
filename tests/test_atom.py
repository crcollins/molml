import os
import unittest

import numpy

from molml.atom import Shell
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

if __name__ == '__main__':
    unittest.main()
