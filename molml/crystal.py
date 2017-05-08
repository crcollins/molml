"""
A module to compute molecule based representations.

This module contains a variety of methods to extract features from molecules
based on the entire molecule. All of the methods included here will produce
one vector per molecule input.
"""
import numpy
import scipy

from .base import BaseFeature, InputTypeMixin
from .molecule import CoulombMatrix
from .utils import _radial_iterator


__all__ = ("GenerallizedCrystal", "EwaldSumMatrix", "SineMatrix")


class GenerallizedCrystal(InputTypeMixin, BaseFeature):
    """
    A wrapper around other features to facilitate faking crystals.

    This is done by a brute force expansion of atoms in the molecules based on
    a given unit cell. This is highly inefficient, but it does set a baseline.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    transformer : BaseFeature, default=None
        The transformer that will be used once the atoms have been expanded
        into the crystal.

    radius : float, default=None
        The cutoff radius for including unit cells in angstroms.

    units : list or int, default=None
        The number of unit cells to include for each axis (if this is an int,
        then it is the same for all).

    References
    ----------
    Faber, F.; Lindmaa, A; von Lilienfeld, O. A.; Armiento, R. Crystal
    Structure Representations for Machine Learning Models of Formation
    Energies. arXiv:1503.07406
    """
    ATTRIBUTES = None
    LABELS = None

    def __init__(self, input_type=None, n_jobs=1, transformer=None,
                 radius=None, units=None):
        super(GenerallizedCrystal, self).__init__(input_type=input_type,
                                                  n_jobs=n_jobs)
        self.check_transformer(transformer)
        self.transformer = transformer
        if radius is not None and units is not None:
            msg = "`radius` and `units` can not be set at the same time."
            raise ValueError(msg)
        self.radius = radius
        self.units = units

        self._old_convert_input = self.transformer.convert_input
        self.transformer.convert_input = self.convert_input

    def convert_input(self, X):
        temp = self._old_convert_input(X)
        temp.fill_in_crystal(radius=self.radius, units=self.units)
        return temp

    def fit(self, X, y=None):
        return self.transformer.fit(X)

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        return self.transformer.transform(X)


class EwaldSumMatrix(CoulombMatrix):
    r"""
    In this construction, we use a similar form to the Ewald sum of breaking
    the interaction into three parts and adding them together.

    The interaction between two atoms is defined as follows

    .. math::
        x_{ij} = x_{ij}^{(r)} + x_{ij}^{(m)} + x_{ij}^0.


    The components are defined as follows

    .. math::

        x_{ij}^{(r)} = Z_i Z_j
        \sum_L \frac{\text{erfc}(\alpha \| r_i - r_j + L \|_2)}
                                        {\| r_i - r_j + L \|_2}

        x_{ij}^{(m)} = \frac{Z_i Z_j}{\pi V}
                            \sum_G \frac{e^{-\|G\|_2^2 / (2 \alpha)^2}}
                            {\|G\|_2^2} \cos(G \cdot (r_i - r_j))

        x_{ij}^0 = -(Z_i^2 + Z_j^2) \frac{\alpha}{\sqrt{\pi}} -
                    (Z_i + Z_j)^2 \frac{\pi}{2 V \alpha^2}

        x_{ii} = -Z_i^2 \frac{\alpha}{\sqrt{\pi}} -
                  Z_i^2 \frac{\pi}{2 V \alpha^2}

        \alpha = \sqrt{\pi} \left(\frac{0.01 M}{V}\right)^{1/6}

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    sort : bool, default=False
        Specifies whether or not to sort the coulomb matrix based on the
        sum of the rows (same as L1 norm).

    eigen : bool, default=False
        Specifies whether or not to use the eigen spectrum of the coulomb
        matrix rather than the matrix itself. This changes the scaling to be
        linear in the number of atoms.

    Attributes
    ----------
    _max_size : int
        The size of the largest molecule in the fit molecules by number of
        atoms.

    References
    ----------
    Faber, F.; Lindmaa, A; von Lilienfeld, O. A.; Armiento, R. Crystal
    Structure Representations for Machine Learning Models of Formation
    Energies. arXiv:1503.07406
    """
    ATTRIBUTES = ("_max_size", )
    LABELS = None

    def __init__(self, input_type='list', n_jobs=1, L_max=10, G_max=10,
                 sort=False, eigen=False):
        super(EwaldSumMatrix, self).__init__(input_type=input_type,
                                             n_jobs=n_jobs)
        self._max_size = None
        self.L_max = L_max
        self.G_max = G_max
        self.sort = sort
        self.eigen = eigen

    def _para_transform(self, X):
        """
        A single instance of the transform procedure.

        This is formulated in a way that the transformations can be done
        completely parallel with map.

        Parameters
        ----------
        X : object
            An object to use for the transform

        Returns
        -------
        value : array
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.

        ValueError
            If the size of the transforming molecules are larger than the fit.
        """
        self.check_fit()

        data = self.convert_input(X)
        if len(data.numbers) > self._max_size:
            msg = "The fit molecules (%d) were not as large as the ones that"
            msg += " are being transformed (%d)."
            raise ValueError(msg % (self._max_size, len(data.numbers)))

        ZZ = numpy.outer(data.numbers, data.numbers)
        numpy.fill_diagonal(ZZ, 0)
        rr = data.coords[:, None] - data.coords
        erfc = scipy.special.erfc
        norm = numpy.linalg.norm
        B = data.unit_cell
        Binv = 2 * numpy.pi * numpy.linalg.inv(B)
        V = numpy.linalg.det(B)

        alpha = numpy.pi ** 0.5 * (0.01 * len(data.numbers) / V) ** (1./6)

        # Short range interactions
        xr = numpy.zeros(ZZ.shape)
        for L in _radial_iterator(B, self.L_max):
            # TODO: optimize symmetry
            temp = norm(rr + L, axis=2)
            with numpy.errstate(divide='ignore'):
                xr += erfc(alpha * temp) / temp
        with numpy.errstate(invalid='ignore'):
            xr *= ZZ

        # Long range interactions
        xm = numpy.zeros(ZZ.shape)
        for G in _radial_iterator(Binv, self.G_max):
            # TODO: optimize symmetry
            temp = norm(G) ** 2
            if not temp:
                continue
            first = numpy.exp(-temp / (2*alpha) ** 2) / temp
            second = numpy.cos(rr.dot(G))
            xm += first * second
        xm *= 1. / (numpy.pi * V) * ZZ

        # Constant
        num2 = data.numbers ** 2
        factor = numpy.pi / (2 * V * alpha ** 2)
        x0 = numpy.add.outer(num2, num2) * alpha/numpy.pi - ZZ ** 2 * factor

        # Penultimate
        values = xr + xm + x0

        # Final
        xii = alpha / numpy.sqrt(numpy.pi) + factor
        xii *= -num2
        numpy.fill_diagonal(values, xii)

        if self.sort:
            order = numpy.argsort(values.sum(0))[::-1]
            values = values[order, :][:, order]

        if self.eigen:
            values = numpy.linalg.eig(values)[0]

        padding_difference = self._max_size - len(data.numbers)
        values = numpy.pad(values,
                           (0, padding_difference),
                           mode="constant")
        return values.reshape(-1)


class SineMatrix(CoulombMatrix):
    r"""
    A molecular descriptor based on Coulomb interactions.

    This is a feature that uses a Coulomb-like interaction between all atoms
    in the molecule to generate a matrix that is then vectorized.

    .. math::

        C_{ij} = \begin{cases}
        Z_i Z_j \Phi(r_i, r_j) & i \neq j \\
                          0.5 Z_i^{2.4} & i = j
        \end{cases}

    Where :math:`\Phi(r_i, r_j)`

    .. math::

        \| B \cdot \sum_{k={x,y,z}} \hat e_k \sin^2 \left[
                    \pi \hat e_k B^{-1} \cdot (r_i - r_j) \right] \|_2^{-1}

    and :math:`B` is a matrix of the lattice basis vectors.


    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    sort : bool, default=False
        Specifies whether or not to sort the coulomb matrix based on the
        sum of the rows (same as L1 norm).

    eigen : bool, default=False
        Specifies whether or not to use the eigen spectrum of the coulomb
        matrix rather than the matrix itself. This changes the scaling to be
        linear in the number of atoms.

    Attributes
    ----------
    _max_size : int
        The size of the largest molecule in the fit molecules by number of
        atoms.

    References
    ----------
    Faber, F.; Lindmaa, A; von Lilienfeld, O. A.; Armiento, R. Crystal
    Structure Representations for Machine Learning Models of Formation
    Energies. arXiv:1503.07406
    """
    ATTRIBUTES = ("_max_size", )
    LABELS = None

    def __init__(self, input_type='list', n_jobs=1, sort=False, eigen=False):
        super(SineMatrix, self).__init__(input_type=input_type,
                                         n_jobs=n_jobs)
        self._max_size = None
        self.sort = sort
        self.eigen = eigen

    def _para_transform(self, X):
        """
        A single instance of the transform procedure.

        This is formulated in a way that the transformations can be done
        completely parallel with map.

        Parameters
        ----------
        X : object
            An object to use for the transform

        Returns
        -------
        value : array
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.

        ValueError
            If the size of the transforming molecules are larger than the fit.
        """
        self.check_fit()

        data = self.convert_input(X)
        if len(data.numbers) > self._max_size:
            msg = "The fit molecules (%d) were not as large as the ones that"
            msg += " are being transformed (%d)."
            raise ValueError(msg % (self._max_size, len(data.numbers)))

        padding_difference = self._max_size - len(data.numbers)

        # Standard parts
        ZZ = numpy.outer(data.numbers, data.numbers).astype(float)
        rr = data.coords[:, None] - data.coords

        # Compute phi
        B = data.unit_cell
        Binv = numpy.linalg.inv(B)
        inner = numpy.tensordot(Binv, rr, [[1], [2]])
        inner *= numpy.pi
        numpy.sin(inner, inner)
        numpy.square(inner, inner)
        full = numpy.tensordot(B, inner, [[1], [0]])
        phi = numpy.linalg.norm(full, axis=0)
        with numpy.errstate(divide='ignore'):
            numpy.power(phi, -1, phi)

        # Final
        ZZ *= phi
        diag = 0.5 * data.numbers ** 2.4
        numpy.fill_diagonal(ZZ, diag)
        values = ZZ

        if self.sort:
            order = numpy.argsort(values.sum(0))[::-1]
            values = values[order, :][:, order]

        if self.eigen:
            values = numpy.linalg.eig(values)[0]

        values = numpy.pad(values,
                           (0, padding_difference),
                           mode="constant")
        return values.reshape(-1)
