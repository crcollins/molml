"""
A module to compute molecule based representations.

This module contains a variety of methods to extract features from molecules
based on the entire molecule. All of the methods included here will produce
one vector per molecule input.
"""
from builtins import range
from itertools import product

import numpy
import scipy

from .molecule import CoulombMatrix


__all__ = ("EwaldSumMatrix", "SineMatrix")


class EwaldSumMatrix(CoulombMatrix):
    r"""
    A molecular descriptor based on Coulomb interactions.

    This is a feature that uses a Coulomb-like interaction between all atoms
    in the molecule to generate a matrix that is then vectorized.

    .. math::

        C_{ij} = \begin{cases}
        \frac{Z_i Z_j}{\| r_i - r_j \|} & i \neq j \\
                          0.5 Z_i^{2.4} & i = j
        \end{cases}


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
    https://arxiv.org/pdf/1503.07406v1.pdf
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

    def _radial_iterator(X, X_max):
        # We will just approximate the upper bound
        maxval = X.max()
        steps = int(X_max / maxval) + 1

        others = [[1], [1], [1]]
        for i in range(steps):
            if i:
                others[0] = [1, -1]
            temp_x = X[:, 0] * i
            if numpy.linalg.norm(temp_x) > X_max:
                continue

            for j in range(steps):
                if j:
                    others[1] = [1, -1]
                temp_y = temp_x + X[:, 1] * j
                if numpy.linalg.norm(temp_y) > X_max:
                    continue

                for k in range(steps):
                    if k:
                        others[2] = [1, -1]
                    temp_z = temp_y + X[:, 2] * k
                    if numpy.linalg.norm(temp_z) > X_max:
                        continue

                    for group in product(*others):
                        a = numpy.array(group)
                        yield temp_z * a

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
        for L in self._radial_iterator(B, self.L_max):
            # TODO: optimize symmetry
            temp = norm(rr + L, axis=2)
            xr += erfc(alpha * temp) / temp
        xr *= ZZ

        # Long range interactions
        xm = numpy.zeros(ZZ.shape)
        for G in self._radial_iterator(Binv, self.G_max):
            # TODO: optimize symmetry
            temp = norm(G) ** 2
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
        numpy.set_diagonal(values, xii)

        if self.sort:
            order = numpy.argsort(values.sum(0))[::-1]
            values = values[order, :][:, order]

        if self.eigen:
            values = numpy.linalg.eig(values)[0]

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

    Where \Phi(r_i, r_j)

    .. math::

        \| B \cdot \sum_{k={x,y,z}} \hat e_k \sin^2 \left[
                    \pi \hat e_k B^{-1} \cdot (r_i - r_j) \right] \|_2^{-1}

    and B is a matrix of the lattice basis vectors.


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
    https://arxiv.org/pdf/1503.07406v1.pdf
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
        ZZ = numpy.outer(data.numbers, data.numbers)
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
        numpy.power(phi, -1)

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
