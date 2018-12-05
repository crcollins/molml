"""
A module to compute atom based representations.

This module contains a variety of methods to extract features from molecules
based on the atoms in the molecule. This means that every molecule will
result in an array of values (n_atoms, n_features).
"""
from builtins import range
from itertools import product

import numpy
from scipy.spatial.distance import cdist

from .base import BaseFeature, SetMergeMixin, EncodedFeature
from .utils import get_depth_threshold_mask_connections, get_coulomb_matrix
from .utils import get_element_pairs, cosine_decay, get_angles
from .utils import get_index_mapping


__all__ = ("Shell", "LocalEncodedBond", "LocalEncodedAngle",
           "LocalCoulombMatrix", "BehlerParrinello")


class Shell(SetMergeMixin, BaseFeature):
    """
    A feature that counts the number of elements in a distance shell from the
    starting atom. This is similar to the features developed in Qu et. al.
    with the exception that it is atom-based rather than bond-based.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    depth : int, default=1
        The length of the atom chains to generate for connections

    use_coordination : boolean, default=False
        Specifies whether or not to use the coordination number of the atoms
        (C1 vs C2 vs C3 vs C4).

    add_unknown : boolean, default=False
        Specifies whether or not to include an extra UNKNOWN count in the
        feature vector.

    Attributes
    ----------
    _elements : tuple
        All the elements/types that are in the fit molecules.

    References
    ----------
    Qu, X.; Latino, D. A.; Aires-de Sousa, J. A Big Data Approach to the
    Ultra-fast Prediction of DFT-calculated Bond Energies. J. Cheminf. 2013,
    5, 34.
    """
    ATTRIBUTES = ("_elements", )
    LABELS = ("_elements", )

    def __init__(self, input_type='list', n_jobs=1, depth=1,
                 use_coordination=False, add_unknown=False):
        super(Shell, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self.depth = depth
        self.use_coordination = use_coordination
        self.add_unknown = add_unknown
        self._elements = None

    def _loop_depth(self, start, connections):
        """
        Loop over the depth number expanding chains. Only keep the elements
        in the last shell.

        Parameters
        ----------
        start : int
            The index of the atom to start the search from

        connections : dict, key->list of keys
            A dictonary edge table with all the bidirectional connections

        Returns
        -------
        limit : list
            A list of index values that are at the given depth.
        """
        # This is just a slightly modified breadth-first search
        visited = {start: 1}
        frontier = [start]

        limit = []
        while len(frontier):
            node = frontier.pop(0)
            prev_depth = visited[node]
            if prev_depth >= self.depth:
                limit.append(node)
                continue

            for x in connections[node]:
                if x in visited:
                    continue
                visited[x] = prev_depth + 1
                frontier.append(x)
        return limit

    def _tally_limits(self, limits, elements, connections=None):
        """
        Tally limit values and return a dictonary with counts of the types.

        Parameters
        ----------
        limits : list
            All of the elements in the molecule at an end point

        nodes : list
            All of the element labels of the atoms

        connections : dict, key->list of keys
            A dictonary edge table with all the bidirectional connections

        Returns
        -------
        counts : dict, element->int
            Totals of the number of each type of element
        """
        counts = {}
        for x in limits:
            ele = elements[x]
            if self.use_coordination:
                ele += str(len(connections[x]))
            if ele not in counts:
                counts[ele] = 0
            counts[ele] += 1
        return counts

    def _para_fit(self, X):
        """
        A single instance of the fit procedure.

        This is formulated in a way that the fits can be done completely
        parallel in a map/reduce fashion.

        Parameters
        ----------
        X : object
            An object to use for the fit

        Returns
        -------
        value : set
            All the elements in the molecule
        """
        data = self.convert_input(X)
        # This is just a cheap way to approximate the actual value
        elements = data.elements
        connections = data.connections
        if self.use_coordination:
            elements = [ele + str(len(connections[i])) for i, ele in
                        enumerate(elements)]
        return set(elements)

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
        value : list, shape=(n_atoms, len(self._elements))
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.
        """
        self.check_fit()
        data = self.convert_input(X)

        vectors = []
        for atom in range(len(data.elements)):
            limits = self._loop_depth(atom, data.connections)
            tallies = self._tally_limits(limits, data.elements,
                                         data.connections)
            vec = [tallies.get(x, 0) for x in self._elements]
            if self.add_unknown:
                unknown = 0
                for key, value in tallies.items():
                    if key not in self._elements:
                        unknown += value
                vec.append(unknown)
            vectors.append(vec)
        return vectors


class LocalEncodedBond(SetMergeMixin, EncodedFeature):
    """
    A smoothed histogram of atomic distances.

    This is a method to generallize the idea of bond counting. Instead of
    seeing bonds as a discrete count that is thresholded at a given length,
    they are seen as general distance histograms. This is supplemented with
    smoothing functions. This is a slight modification of the EncodedBond
    to use with atoms.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    segments : int, default=100
        The number of bins/segments to use when generating the histogram.

    smoothing : string or callable, default='norm'
        A string or callable to use to smooth the histogram values. If a
        callable is given, it must take just a single argument that is a float.
        For a list of supported default functions look at SMOOTHING_FUNCTIONS.

    start : float, default=0.2
        The starting point for the histgram sampling in angstroms.

    end : float, default=6.0
        The ending point for the histogram sampling in angstroms.

    slope : float, default=20.
        A parameter to tune the smoothing values. This is applied as a
        multiplication before calling the smoothing function.

    min_depth : int, default=0
        A parameter to set the minimum geodesic distance to include in the
        interactions. A value of np.inf signifies including only intermolecular
        interactions.

    max_depth : int, default=0
        A parameter to set the maximum geodesic distance to include in the
        interactions. A value of 0 signifies that all interactions are
        included.

    spacing : string or callable, default='linear'
        The histogram interval spacing type. Must be one of ("linear",
        "inverse", or "log"). Linear spacing is normal spacing. Inverse takes
        and evaluates the distances as 1/r and the start and end points are
        1/x. For log spacing, the distances are evaluated as numpy.log(r)
        and the start and end points are numpy.log(x). If the value is
        callable, then it should take a float or vector of floats and return
        a similar mapping like the other methods.

    form : int, default=1
        The histogram splitting style to use. This value changes the scaling
        of this method to be O(E) or O(1) for 1 or 0 respectively (where E is
        the number of elements).

    add_unknown : boolean, default=False
        Specifies whether or not to include an extra UNKNOWN count in the
        feature vector.

    Attributes
    ----------
    _elements : tuple
        A tuple of all the elements in the fit molecules.
    """
    ATTRIBUTES = ("_elements", )
    LABELS = (("get_encoded_labels", "_elements"), )

    def __init__(self, input_type='list', n_jobs=1, segments=100,
                 smoothing='norm', start=0.2, end=6.0, slope=20., min_depth=0,
                 max_depth=0, spacing='linear', form=1, add_unknown=False):
        super(LocalEncodedBond, self).__init__(input_type=input_type,
                                               n_jobs=n_jobs,
                                               segments=segments,
                                               smoothing=smoothing,
                                               start=start,
                                               end=end,
                                               slope=slope,
                                               spacing=spacing)
        self._elements = None
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.form = form
        self.add_unknown = add_unknown

    def _para_fit(self, X):
        """
        A single instance of the fit procedure.

        This is formulated in a way that the fits can be done completely
        parallel in a map/reduce fashion.

        Parameters
        ----------
        X : object
            An object to use for the fit

        Returns
        -------
        value : set
            All the element pairs in the molecule
        """
        data = self.convert_input(X)
        # This is just a cheap way to approximate the actual value
        return set(data.elements)

    def _iterator(self, data, get_index, both=None):
        mat = get_depth_threshold_mask_connections(data.connections,
                                                   min_depth=self.min_depth,
                                                   max_depth=self.max_depth)
        distances = cdist(data.coords, data.coords)
        for i, ele1 in enumerate(data.elements):
            for j, ele2 in enumerate(data.elements):
                if i == j or not mat[i, j]:
                    continue
                try:
                    idx = i, get_index(ele2)
                except KeyError:
                    idx = None
                yield idx, distances[i, j], 1.

    def _para_transform(self, X, y=None):
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
        value : array, shape=(n_atoms, len(self._elements) * self.segments)
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.
        """
        self.check_fit()
        data = self.convert_input(X)
        get_index, length, _ = get_index_mapping(self._elements, self.form,
                                                 self.add_unknown)
        iterator = self._iterator(data, get_index)
        return self.encode_values(iterator, (len(data.elements), length),
                                  saved_lengths=1)


class LocalEncodedAngle(SetMergeMixin, EncodedFeature):
    r"""
    A smoothed histogram of atomic angles.

    This method is similar to EncodedBond but for angles in molecules. This is
    done by enumerating triplets of atoms and computing the angle between
    them. The bins are thing smoothed with smoothing functions. This is a
    slight modification of the EncodedAngle to work with single atoms at a
    time. This sets the vertex of the angle to be the atom being examined.

    Note: The angles used are 0 to :math:`\pi`.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    segments : int, default=100
        The number of bins/segments to use when generating the histogram.

    smoothing : string or callable, default='norm'
        A string or callable to use to smooth the histogram values. If a
        callable is given, it must take just a single argument that is a float.
        For a list of supported default functions look at SMOOTHING_FUNCTIONS.

    slope : float, default=20.
        A parameter to tune the smoothing values. This is applied as a
        multiplication before calling the smoothing function.

    min_depth : int, default=0
        A parameter to set the minimum geodesic distance to include in the
        interactions. A value of np.inf signifies including only intermolecular
        interactions.

    max_depth : int, default=0
        A parameter to set the maximum geodesic distance to include in the
        interactions. A value of 0 signifies that all interactions are
        included.

    form : int, default=2
        The histogram splitting style to use. This value changes the scaling
        of this method to be O(E^2), O(E), or O(1) for 2, 1, or 0 respectively
        (where E is the number of elements).

    add_unknown : boolean, default=False
        Specifies whether or not to include an extra UNKNOWN count in the
        feature vector.

    Attributes
    ----------
    _pairs : tuple
        A tuple of all the element pairs in the fit molecules.
    """
    ATTRIBUTES = ("_pairs", )
    LABELS = (("get_encoded_labels", "_pairs"), )

    def __init__(self, input_type='list', n_jobs=1, segments=100,
                 smoothing='norm', slope=20., min_depth=0, max_depth=0,
                 r_cut=6., form=2, add_unknown=False):
        super(LocalEncodedAngle, self).__init__(input_type=input_type,
                                                n_jobs=n_jobs,
                                                segments=segments,
                                                smoothing=smoothing,
                                                slope=slope,
                                                start=0.,
                                                end=numpy.pi)
        self._pairs = None
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.r_cut = r_cut
        self.form = form
        self.add_unknown = add_unknown

    def f_c(self, R):
        return cosine_decay(R, r_cut=self.r_cut)

    def _para_fit(self, X):
        """
        A single instance of the fit procedure.

        This is formulated in a way that the fits can be done completely
        parallel in a map/reduce fashion.

        Parameters
        ----------
        X : object
            An object to use for the fit

        Returns
        -------
        value : set
            All the element pairs in the molecule
        """
        data = self.convert_input(X)
        # This is just a cheap way to approximate the actual value
        return get_element_pairs(data.elements)

    def _iterator(self, data, get_index, both):
        mat = get_depth_threshold_mask_connections(data.connections,
                                                   min_depth=self.min_depth,
                                                   max_depth=self.max_depth)

        distances = cdist(data.coords, data.coords)
        f_c = self.f_c(distances)
        angles = get_angles(data.coords)
        for i, ele1 in enumerate(data.elements):
            for j, ele2 in enumerate(data.elements):
                if i == j or not mat[i, j]:
                    continue
                if not f_c[i, j]:
                    continue
                for k, ele3 in enumerate(data.elements):
                    if j == k or not mat[j, k]:
                        continue
                    if i > k and not both:
                        continue
                    if not f_c[i, k] or not f_c[j, k]:
                        continue

                    eles = ele1, ele3
                    try:
                        idx = j, get_index(eles)
                    except KeyError:
                        idx = None
                    F = f_c[i, j] * f_c[j, k] * f_c[i, k]
                    yield idx, angles[i, j, k], F

    def _para_transform(self, X, y=None):
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
        value : array, shape=(n_atoms, len(self._pairs) * self.segments)
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.
        """
        self.check_fit()
        data = self.convert_input(X)
        get_index, length, both = get_index_mapping(self._pairs, self.form,
                                                    self.add_unknown)

        iterator = self._iterator(data, get_index, both)
        return self.encode_values(iterator, (len(data.elements), length),
                                  saved_lengths=1)


class LocalCoulombMatrix(BaseFeature):
    r"""
    An implementation of the Coulomb Matrix where only the local atom
    environment is used by using a cutoff radius.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    max_occupancy : int, default=4
        The maximum number of atoms to be included the in local environment.

    r_cut : float, default=6
        The maximum distance allowed for atoms to be considered local to the
        "central atom".

    alpha : number, default=6
        Some value to exponentiate the distance in the coulomb matrix.

    use_reduced : bool, default=False
        This setting uses only the first row of the local coulomb matrix and
        the diagonal. This reduces the feature from scaling as
        O(max_occupancy ** 2) to just O(max_occupancy).

    use_decay : bool, default=False
        This setting defines an extra decay for the values as they get futher
        away from the "central atom". This is to alleviate issues the arise as
        atoms enter or leave the cutoff radius.

        .. math::

            M_{ij} = \begin{cases}
            \frac{Z_{p_i} Z_{p_j}}{(\| R_{p_1} - R_{p_i} \|_2
                                  + \| R_{p_1} - R_{p_j} \|_2
                                  + \| R_{p_i} - R_{p_j} \|_2 )^{\alpha}},
                                  & i \neq j \\
                             0.5 Z_{p_i}^{2.4} & i = j
            \end{cases}

    References
    ----------
    Barker, J.; Bulin, J.;  Hamaekers, J.; Mathias, S. Localized Coulomb
    Descriptors for the Gaussian Approximation Potential. arXiv 1611.05126
    """
    ATTRIBUTES = None
    LABELS = (('get_local_coulomb_labels', None), )

    def __init__(self, input_type='list', n_jobs=1, max_occupancy=4, r_cut=10.,
                 alpha=6, use_reduced=False, use_decay=False):
        super(LocalCoulombMatrix, self).__init__(input_type=input_type,
                                                 n_jobs=n_jobs)
        self.max_occupancy = max_occupancy
        self.r_cut = r_cut
        self.alpha = alpha
        self.use_reduced = use_reduced
        self.use_decay = use_decay

    def fit(self, X, y=None):
        """No fitting is required because it is defined by the parameters."""
        return self

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
        """
        data = self.convert_input(X)
        dist = cdist(data.coords, data.coords)

        numbers = numpy.array(data.numbers)
        coords = numpy.array(data.coords)

        vectors = []
        for i in range(len(numbers)):
            nearest = numpy.where(dist[i, :] < self.r_cut)
            ordering = numpy.argsort(dist[i, :][nearest])
            # Add 1 to offset for the start value
            local_atoms = ordering[:self.max_occupancy + 1]
            mat = get_coulomb_matrix(numbers[local_atoms],
                                     coords[local_atoms],
                                     alpha=self.alpha,
                                     use_decay=self.use_decay)
            # Take away 1 for the start value
            n = self.max_occupancy - (len(local_atoms) - 1)
            mat = numpy.pad(mat, ((0, n), (0, n)), "constant")
            norm_vals = numpy.linalg.norm(mat, axis=0)
            norm_vals[0] = numpy.inf
            sorting = numpy.argsort(norm_vals)[::-1]
            if self.use_reduced:
                # skip the first value in the diag because it is already in
                # the first row
                diag = numpy.diag(mat)[1:].tolist()
                vectors.append(mat[sorting[0]].tolist() + diag)
            else:
                vectors.append(mat[sorting].flatten())
        return numpy.array(vectors)

    def get_local_coulomb_labels(self):
        idxs = []

        vals = list(range(self.max_occupancy + 1))
        if self.use_reduced:
            for i in vals:
                idxs.append((0, i))
            for i in vals[1:]:
                idxs.append((i, i))
        else:
            for pair in product(vals, vals):
                idxs.append(pair)
        return ['local-coul_%d-%d' % (i, j) for i, j in idxs]


class BehlerParrinello(SetMergeMixin, BaseFeature):
    """
    An implementation of the descriptors used in Behler-Parrinello Neural
    Networks.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    r_cut : float, default=6.
        The maximum distance allowed for atoms to be considered local to the
        "central atom".

    r_s : float, default=1.0
        An offset parameter for computing gaussian values between pairwise
        distances.

    eta : float, default=1.0
        A decay parameter for the gaussian distances.

    lambda_ : float, default=1.0
        This value sets the orientation of the cosine function for the angles.
        It should only take values in {-1., 1.}.

    zeta : float, default=1.0
        A decay parameter for the angular terms.

    Attributes
    ----------
    _elements : tuple
        A set of all the elements in the molecules.

    _element_pairs : tuple
        A set of all the element pairs in the molecules.

    References
    ----------
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401.
    """
    ATTRIBUTES = ("_elements", "_element_pairs")
    LABELS = ("_elements", ("get_chain_labels", "_element_pairs"))

    def __init__(self, input_type='list', n_jobs=1, r_cut=6.0, r_s=1., eta=1.,
                 lambda_=1., zeta=1.):
        super(BehlerParrinello, self).__init__(input_type=input_type,
                                               n_jobs=n_jobs)
        self.r_cut = r_cut
        self.r_s = r_s
        self.eta = eta
        self.lambda_ = lambda_
        self.zeta = zeta
        self._elements = None
        self._element_pairs = None

    def _para_fit(self, X):
        """
        A single instance of the fit procedure.

        This is formulated in a way that the fits can be done completely
        parallel in a map/reduce fashion.

        Parameters
        ----------
        X : object
            An object to use for the fit

        Returns
        -------
        unique_elements : set
            A set of all the elements in the molecule

        pairs : list
            A list of all the element pairs in the molecule
        """
        data = self.convert_input(X)
        # This is just a cheap way to approximate the actual value
        unique_elements = set(data.elements)
        pairs = get_element_pairs(data.elements)
        return unique_elements, pairs

    def f_c(self, R):
        return cosine_decay(R, r_cut=self.r_cut)

    def g_1(self, R, elements):
        r"""
        A radial symmetry function.

        .. math::

            G^1_i = \sum_{j \neq i} \exp(- \eta (R_{ij} - R_s)^2) f_c(R_{ij})

        Parameters
        ----------
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist)

        Returns
        -------
        total : array, shape=(N_atoms, N_elements)
            The atom-wise g_1 evaluations.
        """
        values = numpy.exp(-self.eta * (R - self.r_s) ** 2) * self.f_c(R)
        numpy.fill_diagonal(values, 0)

        elements = numpy.array(elements)

        totals = []
        for ele in self._elements:
            idxs = numpy.where(elements == ele)[0]
            total = values[:, idxs].sum(1)
            totals.append(total)
        return numpy.array(totals).T

    def g_2(self, Theta, R, elements):
        r"""
        An angular symmetry function.

        .. math::

            G^2_i = 2^{1-\zeta} \sum_{i,k \neq i}
                        (1 + \lambda \cos(\Theta_{ijk}))^\zeta
                        \exp(-\eta (R_{ij}^2 + R_{ik}^2 + R_{jk}^2))
                        f_c(R_{ij}) f_c(R_{ik}) f_c(R_{jk})


        This function needs to be optimized.

        Parameters
        ----------
        Theta : array, shape=(N_atoms, N_atoms, N_atoms)
            An array of triplet angles.

        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist).

        elements : list
            A list of all the elements in the molecule.

        Returns
        -------
        total : array, shape=(N_atoms, len(self._element_pairs))
            The atom-wise g_2 evaluations.
        """
        F_c_R = self.f_c(R)

        R2 = self.eta * R ** 2
        new_Theta = (1 + self.lambda_ * Theta) ** self.zeta

        get_index, length, _ = get_index_mapping(self._element_pairs, 2, False)

        n = R.shape[0]
        values = numpy.zeros((n, length))
        for i in range(n):
            for j in range(n):
                if i == j or not F_c_R[i, j]:
                    continue
                ele1 = elements[j]

                for k in range(n):
                    if k == i or j == k:
                        continue
                    if not F_c_R[i, k] or not F_c_R[j, k]:
                        continue
                    ele2 = elements[k]
                    eles = ele1, ele2

                    exp_term = numpy.exp(-(R2[i, j] + R2[i, k] + R2[j, k]))
                    angular_term = new_Theta[i, j, k]
                    radial_cuts = F_c_R[i, j] * F_c_R[i, k] * F_c_R[j, k]
                    temp = angular_term * exp_term * radial_cuts
                    try:
                        values[i, get_index(eles)] += temp
                    except KeyError:
                        pass
        return 2 ** (1 - self.zeta) * values

    def calculate_Theta(self, R_vecs):
        """
        Compute the angular term for all triples of atoms.

        .. math::

            \Theta_{ijk} = (R_{ij} . R_{ik}) / (|R_{ij}| |R_{ik}|)

        Right now this is a fairly naive implementation so this could be
        optimized quite a bit.

        Parameters
        ----------
        R_vecs : array, shape=(N_atoms, 3)
            An array of the Cartesian coordinates of all the atoms

        Returns
        -------
        Theta : array, shape=(N_atoms, N_atoms, N_atoms)
            The angular term for all the atoms given.
        """
        n = R_vecs.shape[0]
        Theta = numpy.zeros((n, n, n))
        for i, Ri in enumerate(R_vecs):
            for j, Rj in enumerate(R_vecs):
                if i == j:
                    continue
                Rij = Ri - Rj
                normRij = numpy.linalg.norm(Rij)
                for k, Rk in enumerate(R_vecs):
                    if i == k or j == k:
                        continue
                    Rik = Ri - Rk
                    normRik = numpy.linalg.norm(Rik)
                    Theta[i, j, k] = numpy.dot(Rij, Rik) / (normRij * normRik)
        return Theta

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
        """
        self.check_fit()

        data = self.convert_input(X)

        coords = numpy.array(data.coords)
        R = cdist(coords, coords)
        Theta = self.calculate_Theta(coords)

        g1 = self.g_1(R, data.elements)
        g2 = self.g_2(Theta, R, data.elements)
        return numpy.hstack([g1, g2])

    def get_chain_labels(self, chains):
        return ['-'.join(x) for x in chains]
