import unittest

import numpy

from molml.atom import Shell, LocalEncodedBond, LocalCoulombMatrix
from molml.atom import LocalEncodedAngle
from molml.atom import BehlerParrinello
from molml.constants import UNKNOWN

from .constants import METHANE, BIG, MID, ALL_DATA

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
        self.assertEqual(a._elements, ('C', 'H', 'N', 'O'))

    def test_fit_use_coordination(self):
        a = Shell(depth=1, use_coordination=True)
        a.fit(ALL_DATA)
        self.assertEqual(a._elements, ('C1', 'C2', 'C3', 'C4', 'H0', 'H1',
                                       'N1', 'N2', 'N3', 'O0', 'O1', 'O2'))

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

    def test_get_labels(self):
        a = Shell()
        a.fit(ALL_DATA)
        expected = ('C', 'H', 'N', 'O')
        self.assertEqual(a.get_labels(), expected)

    def test_get_labels_unknown(self):
        a = Shell(add_unknown=True)
        a.fit(ALL_DATA)
        expected = ('C', 'H', 'N', 'O', UNKNOWN)
        self.assertEqual(a.get_labels(), expected)


class LocalEncodedBondTest(unittest.TestCase):

    def test_fit(self):
        a = LocalEncodedBond()
        a.fit(ALL_DATA)
        self.assertEqual(a._elements, (('C', ), ('H', ), ('N', ), ('O', )))

    def test_transform(self):
        a = LocalEncodedBond()
        a.fit(ALL_DATA)
        m = a.transform(ALL_DATA)
        expected_results = numpy.array([17.068978019300587,
                                        54.629902544876572,
                                        1006.4744899075993])
        mm = numpy.array([x.sum() for x in m])
        self.assertTrue((numpy.allclose(mm, expected_results)))

    def test_small_to_large(self):
        a = LocalEncodedBond()
        a.fit([METHANE])

        # This is a cheap test to prevent needing all the values here
        expected_results = numpy.array([
            0.016125813269,  # mean
            0.065471987297,  # std
            0.,              # min
            0.398807098298,  # max
            29.02646388512,  # sum
        ])
        try:
            m = a.transform([MID])
            val = numpy.array([
                m.mean(),
                m.std(),
                m.min(),
                m.max(),
                m.sum(),
            ])
            numpy.testing.assert_allclose(val, expected_results)
        except AssertionError as e:
            self.fail(e)

    def test_transform_max_depth1(self):
        a = LocalEncodedBond(max_depth=1)
        a.fit(ALL_DATA)
        m = a.transform(ALL_DATA)
        expected_results = numpy.array([6.82758723,
                                        6.82758018,
                                        88.75860423])
        mm = numpy.array([x.sum() for x in m])
        try:
            numpy.testing.assert_allclose(mm, expected_results)
        except AssertionError as e:
            self.fail(e)

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

    def test_add_unknown(self):
        a = LocalEncodedBond(add_unknown=True)
        a.fit([METHANE])
        m = a.transform([MID])
        self.assertEqual(m.shape, (1, 9, 300))

    def test_form(self):
        a = LocalEncodedBond(form=0)
        m = a.fit_transform([METHANE])
        self.assertEqual(m.shape, (1, 5, 100))

    def test_get_labels(self):
        a = LocalEncodedBond(segments=2, start=0., end=1.)
        m = a.fit_transform([METHANE])
        labels = a.get_labels()
        self.assertEqual(m.shape[2], len(labels))
        expected = (
            'C_0.0', 'C_1.0',
            'H_0.0', 'H_1.0',
        )
        self.assertEqual(labels, expected)


class LocalEncodedAngleTest(unittest.TestCase):

    def test_fit(self):
        a = LocalEncodedAngle()
        a.fit(ALL_DATA)
        expected = (('C', 'C'), ('C', 'H'), ('C', 'N'), ('C', 'O'),
                    ('H', 'H'), ('H', 'N'), ('H', 'O'), ('N', 'N'),
                    ('N', 'O'), ('O', 'O'))
        self.assertEqual(a._pairs, expected)

    def test_transform(self):
        a = LocalEncodedAngle()
        a.fit(ALL_DATA)
        m = a.transform([METHANE, MID])
        expected_results = numpy.array([42.968775,
                                        53.28433])
        mm = numpy.array([x.sum() for x in m])
        try:
            numpy.testing.assert_allclose(mm, expected_results)
        except AssertionError as e:
            self.fail(e)

    def test_small_to_large(self):
        a = LocalEncodedAngle()
        a.fit([METHANE])

        # This is a cheap test to prevent needing all the values here
        expected_results = numpy.array([
            0.005350052647,  # mean
            0.036552192752,  # std
            0.,              # min
            0.614984152986,  # max
            9.630094765591,  # sum
        ])
        try:
            m = a.transform([MID])
            val = numpy.array([
                m.mean(),
                m.std(),
                m.min(),
                m.max(),
                m.sum(),
            ])
            numpy.testing.assert_allclose(val, expected_results)
        except AssertionError as e:
            self.fail(e)

    def test_transform_max_depth1(self):
        a = LocalEncodedAngle(max_depth=1)
        a.fit(ALL_DATA)
        m = a.transform(ALL_DATA)
        expected_results = numpy.array([13.078022,
                                        7.028573,
                                        146.255683])
        mm = numpy.array([x.sum() for x in m])
        try:
            numpy.testing.assert_allclose(mm, expected_results)
        except AssertionError as e:
            self.fail(e)

    def test_transform_before_fit(self):
        a = LocalEncodedAngle()
        with self.assertRaises(ValueError):
            a.transform(ALL_DATA)

    def test_transform_invalid_smoothing(self):
        a = LocalEncodedAngle(smoothing='not real"')
        with self.assertRaises(KeyError):
            a.fit_transform(ALL_DATA)

    def test_add_unknown(self):
        a = LocalEncodedAngle(add_unknown=True)
        a.fit([METHANE])
        m = a.transform([MID])
        self.assertEqual(m.shape, (1, 9, 300))

    def test_form1(self):
        a = LocalEncodedAngle(form=1)
        m = a.fit_transform([METHANE])
        self.assertEqual(m.shape, (1, 5, 200))

    def test_form0(self):
        a = LocalEncodedAngle(form=0)
        m = a.fit_transform([METHANE])
        self.assertEqual(m.shape, (1, 5, 100))

    def test_get_labels(self):
        a = LocalEncodedAngle(segments=2)
        m = a.fit_transform([METHANE])
        labels = a.get_labels()
        self.assertEqual(m.shape[2], len(labels))
        expected = (
            'C-H_0.0', 'C-H_3.14159',
            'H-H_0.0', 'H-H_3.14159',
        )
        self.assertEqual(labels, expected)


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
                [36.8581052, 3.53894587, 0.5],
                [0.5, 3.53879657, 36.8581052],
                [0.5, 3.53894587, 36.8581052],
                [0.5, 3.53844797, 36.8581052],
                [0.5, 3.52229971, 36.8581052],
            ],
            [
                [36.85810519942594, 4.642192257970912, 36.85810519942594],
                [36.85810519942594, 6.483079598556282, 73.51669471981023],
                [73.51669471981023, 8.650089711338763, 73.51669471981023],
                [73.51669471981023, 8.650089711338763, 73.51669471981023],
                [73.51669471981023, 10.698256448660478, 73.51669471981023],
                [73.51669471981023, 10.698256448660478, 73.51669471981023],
                [0.5, 0.04483292, 0.5],
                [0.5, 1., 0.5],
                [0.5, 1., 0.5]
            ]
        ])
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
                [36.8581052, 5.03182436, 0.5],
                [0.5, 5.0317536, 36.8581052],
                [0.5, 5.03182436, 36.8581052],
                [0.5, 5.03158837, 36.8581052],
                [0.5, 5.02392255, 36.8581052],
            ],
            [
                [36.8581052, 18.18762711, 36.8581052],
                [36.8581052, 24.62755378, 73.51669472],
                [73.51669472, 32.84431333, 73.51669472],
                [73.51669472, 32.84431333, 73.51669472],
                [73.51669472, 35.25529211, 73.51669472],
                [73.51669472, 35.25529211, 73.51669472],
                [0.5, 0.35524858, 0.5],
                [0.5, 1., 0.5],
                [0.5, 1., 0.5]
            ]
        ])
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

    def test_get_labels(self):
        a = LocalCoulombMatrix(max_occupancy=1)
        X = a.fit_transform([METHANE])
        labels = a.get_labels()
        self.assertEqual(X.shape[2], len(labels))
        expected = (
            'local-coul_0-0', 'local-coul_0-1',
            'local-coul_1-0', 'local-coul_1-1'
        )
        self.assertEqual(labels, expected)

    def test_get_labels_reduced(self):
        a = LocalCoulombMatrix(max_occupancy=1, use_reduced=True)
        X = a.fit_transform([METHANE])
        labels = a.get_labels()
        self.assertEqual(X.shape[2], len(labels))
        expected = ('local-coul_0-0', 'local-coul_0-1', 'local-coul_1-1')
        self.assertEqual(labels, expected)


class BehlerParrinelloTest(unittest.TestCase):

    def test_fit(self):
        a = BehlerParrinello()
        a.fit(ALL_DATA)
        eles = ('C', 'H', 'N', 'O')
        pairs = (('C', 'C'), ('C', 'H'), ('C', 'N'), ('C', 'O'),
                 ('H', 'H'), ('H', 'N'), ('H', 'O'), ('N', 'N'),
                 ('N', 'O'), ('O', 'O'))
        self.assertEqual(a._elements, eles)
        self.assertEqual(a._element_pairs, pairs)

    def test_transform_before_fit(self):
        a = BehlerParrinello()
        with self.assertRaises(ValueError):
            a.transform(ALL_DATA)

    def test_predict_outside_fit(self):
        a = BehlerParrinello()
        a.fit([METHANE])
        res = a.transform([MID])
        expected = numpy.array([[2.812505e-01,
                                 3.838817e-01,
                                 1.351548e-17,
                                 3.040356e-05]])
        try:
            numpy.testing.assert_array_almost_equal(
                numpy.array([x.mean(0) for x in res]),
                expected)
        except AssertionError as e:
            self.fail(e)

    def test_transform(self):
        a = BehlerParrinello()
        a.fit(ALL_DATA)
        m = a.transform(ALL_DATA)
        expected = numpy.array([
            [7.30122351e-01, 1.76581653e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 2.25151270e-02, 0.00000000e+00, 0.00000000e+00,
             4.39327069e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00],
            [2.81250479e-01, 3.83881730e-01, 0.00000000e+00, 5.18353517e-01,
             1.21609280e-05, 1.35154847e-17, 0.00000000e+00, 3.22304785e-05,
             3.04035608e-05, 0.00000000e+00, 2.17982869e-06, 0.00000000e+00,
             0.00000000e+00, 1.45702791e-05],
            [1.41699320e+00, 4.71095583e-01, 2.07937343e-01, 1.78908050e-01,
             4.35009106e-04, 1.90163675e-03, 3.00682337e-04, 3.86290776e-04,
             4.65644534e-04, 1.19773689e-03, 2.65217272e-04, 2.50781918e-06,
             5.01216130e-06, 4.99291073e-12]
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
            [7.30122351e-01, 1.76581653e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 2.25151270e-02, 0.00000000e+00, 0.00000000e+00,
             4.39327069e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00],
            [2.81250479e-01, 3.83881730e-01, 0.00000000e+00, 5.18353517e-01,
             1.21609280e-05, 1.35154847e-17, 0.00000000e+00, 3.22304785e-05,
             3.04035608e-05, 0.00000000e+00, 2.17982869e-06, 0.00000000e+00,
             0.00000000e+00, 1.45702791e-05],
            [1.41699320e+00, 4.71095583e-01, 2.07937343e-01, 1.78908050e-01,
             4.35009106e-04, 1.90163675e-03, 3.00682337e-04, 3.86290776e-04,
             4.65644534e-04, 1.19773689e-03, 2.65217272e-04, 2.50781918e-06,
             5.01216130e-06, 4.99291073e-12]
        ])
        try:
            numpy.testing.assert_array_almost_equal(
                numpy.array([x.mean(0) for x in m]),
                expected)
        except AssertionError as e:
            self.fail(e)

    def test_get_labels(self):
        a = BehlerParrinello()
        m = a.fit_transform([METHANE])
        labels = a.get_labels()
        self.assertEqual(m[0].shape[1], len(labels))
        expected = ('C', 'H', 'C-H', 'H-H')
        self.assertEqual(labels, expected)


if __name__ == '__main__':
    unittest.main()
