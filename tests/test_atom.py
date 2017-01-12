import os
import unittest

import numpy

from molml.atom import Shell, LocalEncodedBond, LocalCoulombMatrix
from molml.atom import BehlerParrinello
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
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 1, 0, 0]],
    [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1],
     [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 1, 0, 0]],
    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
     [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0],
     [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
     [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0],
     [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
     [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [0, 1, 0, 0]]
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
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0, 0]]
        ])
        self.assertTrue((a.transform([MID]) == expected_results).all())

    def test_transform_depth2(self):
        a = Shell(depth=2)
        a.fit(ALL_DATA)
        expected_results = numpy.array([
            [[0, 4, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
             [1, 0, 0, 0]],
            [[1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0],
             [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0],
             [0, 1, 0, 0]],
            [[1, 0, 1, 1], [2, 1, 0, 0], [2, 1, 0, 0], [2, 1, 0, 0],
             [2, 0, 1, 0], [2, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
             [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 1],
             [3, 0, 0, 0], [2, 1, 0, 0], [2, 0, 0, 1], [2, 0, 1, 0],
             [2, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1],
             [2, 0, 0, 0], [0, 3, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0],
             [1, 0, 0, 0], [2, 0, 0, 1], [3, 0, 0, 0], [3, 0, 0, 0],
             [2, 0, 0, 1], [2, 0, 0, 0], [2, 0, 0, 0], [1, 1, 0, 0],
             [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 2, 0],
             [3, 0, 0, 0], [2, 1, 0, 0], [3, 0, 0, 0], [1, 1, 1, 0],
             [2, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0],
             [0, 0, 1, 0], [0, 0, 1, 0], [2, 0, 0, 0], [1, 1, 0, 0],
             [1, 0, 0, 0]]
        ])
        self.assertTrue((a.transform(ALL_DATA) == expected_results).all())

    def test_transform_depth3(self):
        # This is to test loop backs in the breadth-first search
        a = Shell(depth=3)
        a.fit(ALL_DATA)
        expected_results = numpy.array([
            [[0, 0, 0, 0], [0, 3, 0, 0], [0, 3, 0, 0], [0, 3, 0, 0],
             [0, 3, 0, 0]],
            [[0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0],
             [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[2, 2, 0, 0], [1, 1, 1, 1], [2, 2, 0, 0], [2, 1, 1, 0],
             [4, 1, 0, 0], [3, 0, 0, 1], [2, 0, 0, 0], [2, 0, 0, 0],
             [1, 0, 1, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 1, 0, 0],
             [2, 1, 2, 1], [3, 0, 0, 1], [3, 1, 1, 0], [3, 0, 0, 2],
             [3, 0, 0, 1], [2, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0],
             [2, 3, 0, 0], [1, 0, 0, 0], [0, 2, 0, 1], [0, 2, 0, 1],
             [0, 2, 0, 1], [4, 0, 1, 0], [4, 0, 0, 1], [3, 0, 1, 1],
             [5, 0, 0, 0], [4, 0, 0, 0], [2, 1, 0, 0], [1, 0, 0, 0],
             [1, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 2, 0, 0],
             [2, 1, 2, 1], [4, 0, 0, 0], [2, 2, 1, 0], [3, 0, 0, 0],
             [2, 1, 1, 0], [2, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0],
             [1, 1, 0, 0], [1, 1, 0, 0], [2, 1, 0, 0], [1, 0, 0, 0],
             [1, 0, 0, 0]]
        ])
        self.assertTrue((a.transform(ALL_DATA) == expected_results).all())

    def test_small_to_large_transform(self):
        a = Shell()
        a.fit([METHANE])
        expected = numpy.array([numpy.array(x)[:, :2].tolist()
                                for x in BASE_SHELL])
        self.assertTrue((a.transform(ALL_DATA) == expected).all())

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

    def test_add_unknown(self):
        a = Shell(add_unknown=True)
        a.fit([METHANE])
        temp = []
        for mol in BASE_SHELL:
            inner = []
            for atom in mol:
                inner.append(atom[:2] + [atom[2] + atom[3]])
            temp.append(inner)
        expected = numpy.array(temp)
        self.assertTrue((a.transform(ALL_DATA) == expected).all())


class LocalEncodedBondTest(unittest.TestCase):

    def test_fit(self):
        a = LocalEncodedBond()
        a.fit(ALL_DATA)
        self.assertEqual(a._elements,
                         set(['N', 'C', 'O', 'H']))

    def test_transform(self):
        a = LocalEncodedBond()
        a.fit(ALL_DATA)
        # import pdb; pdb.set_trace()
        m = a.transform(ALL_DATA)
        expected_results = numpy.array([17.068978019300587,
                                        54.629902544876572,
                                        1006.4744899075993])
        mm = numpy.array([x.sum() for x in m])
        self.assertTrue((numpy.allclose(mm, expected_results)))

    def test_transform_max_depth1(self):
        a = LocalEncodedBond(max_depth=1)
        a.fit(ALL_DATA)
        m = a.transform(ALL_DATA)
        expected_results = numpy.array([6.82758723,
                                        6.82758018,
                                        88.75860423])
        mm = numpy.array([x.sum() for x in m])
        self.assertTrue((numpy.allclose(mm, expected_results)))

    def test_transform_before_fit(self):
        a = LocalEncodedBond()
        with self.assertRaises(ValueError):
            a.transform(ALL_DATA)

    def test_transform_invalid_smoothing(self):
        a = LocalEncodedBond(smoothing='not real"')
        with self.assertRaises(KeyError):
            a.fit_transform(ALL_DATA)

    def test_transform_invalid_spacing(self):
        a = LocalEncodedBond(spacing='not real"')
        with self.assertRaises(KeyError):
            a.fit_transform(ALL_DATA)


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


class BehlerParrinelloTest(unittest.TestCase):

    def test_fit(self):
        a = BehlerParrinello()
        a.fit(ALL_DATA)
        eles = set(['H', 'C', 'O', 'N'])
        pairs = set([('H', 'O'), ('C', 'H'), ('H', 'N'), ('C', 'C'),
                     ('H', 'H'), ('O', 'O'), ('C', 'N'), ('N', 'O'),
                     ('C', 'O'), ('N', 'N')])
        self.assertEqual(a._elements, eles)
        self.assertEqual(a._element_pairs, pairs)

    def test_transform_before_fit(self):
        a = BehlerParrinello()
        with self.assertRaises(ValueError):
            a.transform(ALL_DATA)

    def test_transform(self):
        a = BehlerParrinello()
        a.fit(ALL_DATA)
        m = a.transform(ALL_DATA)
        expected = numpy.array([
            [0.7301223510043411, 1.7658165303958306, 0.0, 0.0, 0.0,
             0.003907110564667951, 0.0, 0.0, 0.00036248053943464185,
             0.0, 0.0, 0.0, 0.0, 0.0],
            [0.28125047933241765, 0.3838817298276181, 0.0, 0.5183535170060759,
             2.2662658483043114e-06, 2.146961799899938e-17, 0.0,
             6.224446997510833e-06, 6.621436815009409e-06, 0.0,
             1.3011973400946236e-07, 0.0, 0.0, 2.710341758581579e-06],
            [1.4169932033595218, 0.47109558339134616, 0.20793734297719466,
             0.17890805037295582, 8.551359441496345e-05,
             0.0003343750148296807, 5.237853789060283e-05,
             6.159748056594988e-05, 3.851765263726631e-05,
             0.00021070643998072239, 4.719590988785845e-05,
             4.100380098291933e-07, 8.177583834507374e-07,
             2.461876736177458e-13]
        ])
        try:
            numpy.testing.assert_array_almost_equal(
                numpy.array([x.mean(0) for x in m]),
                expected)
        except AssertionError as e:
            self.fail(e)

    def test_fit_transform(self):
        a = BehlerParrinello()
        m = a.fit_transform(ALL_DATA)
        expected = numpy.array([
            [0.7301223510043411, 1.7658165303958306, 0.0, 0.0, 0.0,
             0.003907110564667951, 0.0, 0.0, 0.00036248053943464185,
             0.0, 0.0, 0.0, 0.0, 0.0],
            [0.28125047933241765, 0.3838817298276181, 0.0, 0.5183535170060759,
             2.2662658483043114e-06, 2.146961799899938e-17, 0.0,
             6.224446997510833e-06, 6.621436815009409e-06, 0.0,
             1.3011973400946236e-07, 0.0, 0.0, 2.710341758581579e-06],
            [1.4169932033595218, 0.47109558339134616, 0.20793734297719466,
             0.17890805037295582, 8.551359441496345e-05,
             0.0003343750148296807, 5.237853789060283e-05,
             6.159748056594988e-05, 3.851765263726631e-05,
             0.00021070643998072239, 4.719590988785845e-05,
             4.100380098291933e-07, 8.177583834507374e-07,
             2.461876736177458e-13]
        ])
        try:
            numpy.testing.assert_array_almost_equal(
                numpy.array([x.mean(0) for x in m]),
                expected)
        except AssertionError as e:
            self.fail(e)


if __name__ == '__main__':
    unittest.main()
