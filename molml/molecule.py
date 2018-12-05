"""
A module to compute molecule based representations.

This module contains a variety of methods to extract features from molecules
based on the entire molecule. All of the methods included here will produce
one vector per molecule input.
"""
from builtins import range
from collections import defaultdict
from itertools import product

import numpy
from scipy.spatial.distance import cdist

from .base import BaseFeature, SetMergeMixin, EncodedFeature
from .utils import get_depth_threshold_mask_connections, get_coulomb_matrix
from .utils import get_element_pairs, cosine_decay, needs_reversal
from .utils import get_index_mapping, get_angles
from .utils import get_graph_distance
from .constants import ELECTRONEGATIVITY, BOND_LENGTHS


__all__ = ("Connectivity", "Autocorrelation", "EncodedAngle", "EncodedBond",
           "CoulombMatrix", "BagOfBonds")


class Connectivity(SetMergeMixin, BaseFeature):
    """
    A collection of feature types based on the connectivity of atoms.

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

    use_bond_order : boolean, default=False
        Specifies whether or not to use bond order information (C-C versus
        C=C). Note: for depth=1, this option does nothing.

    use_coordination : boolean, default=False
        Specifies whether or not to use the coordination number of the atoms
        (C1 vs C2 vs C3 vs C4).

    add_unknown : boolean, default=False
        Specifies whether or not to include an extra UNKNOWN count in the
        feature vector.

    do_tfidf : boolean, default=False
        Apply weighting to counts based on their inverse document (molecule)
        frequency.

    Attributes
    ----------
    _base_chains : tuple, tuples
        All the chains that are in the fit molecules.

    References
    ----------
    Collins, C.; Gordon, G.; von Lilienfeld, O. A.; Yaron, D. Constant Size
    Molecular Descriptors For Use With Machine Learning. arXiv:1701.06649
    """
    ATTRIBUTES = ("_base_chains", "_idf_values")
    LABELS = (("get_chain_labels", "_base_chains"), )

    def __init__(self, input_type='list', n_jobs=1, depth=1,
                 use_bond_order=False, use_coordination=False,
                 add_unknown=False, do_tfidf=False):
        super(Connectivity, self).__init__(input_type=input_type,
                                           n_jobs=n_jobs)
        self.depth = depth
        self.use_bond_order = use_bond_order
        self.use_coordination = use_coordination
        self.add_unknown = add_unknown
        self.do_tfidf = do_tfidf
        self._base_chains = None

        if self.do_tfidf:
            self._idf_values = None
        else:
            self._idf_values = {}

    def _loop_depth(self, connections):
        """
        Loop over the depth number expanding chains.

        Parameters
        ----------
        connections : dict, key->list of keys
            A dictonary edge table with all the bidirectional connections

        Returns
        -------
        chains : list
            A list of key tuples of all the chains in the molecule
        """
        chains = [(x, ) for x in connections]
        for i in range(self.depth - 1):
            chains = self._expand_chains(chains, connections)
        return chains

    def _expand_chains(self, initial, connections):
        """
        Use the connectivity information to add one more atom to each chain.

        Parameters
        ----------
        initial : list
            A list of key tuples of all the chains in the molecule

        connections : dict, key->list of keys
            A dictonary edge table with all the bidirectional connections

        Returns
        -------
        results : list
            A list of index chains that are one index longer than the inputs
            in initial.
        """
        if len(initial) and len(initial[0]) > 1:
            # All of the chains are duplicated and reversed.
            # This is to make the loop simpler when handling both ends of the
            # chain.
            initial = initial + [x[::-1] for x in initial]

        results = {}
        for item in initial:
            # We use the first item because the indexing is easier?
            for x in connections[item[0]]:
                if x in item:
                    continue
                new = (x, ) + item
                if new[0] > new[-1]:
                    new = new[::-1]
                if new not in results:
                    results[new] = 1
        return list(results.keys())

    def _convert_to_bond_order(self, chain, labelled, connections):
        """
        Convert a chain based on elements into one that includes bond order.

        Parameters
        ----------
        chain : tuple
            The atom index values in the chain

        labelled : tuple
            Elements corresponding to the chain indices

        connections : dict, key->list of keys
            A dictonary edge table with all the bidirectional connections

        Returns
        -------
        labelled : list
            The new labelled chain if use_bond_order is set
        """
        if self.use_bond_order and len(labelled) > 1:
            temp = []
            for i, x in enumerate(chain[:-1]):
                idx1 = x
                idx2 = chain[i + 1]
                symbol1 = labelled[i]
                symbol2 = labelled[i + 1]
                temp.append((symbol1, symbol2, connections[idx1][idx2]))
            labelled = temp
        return labelled

    def _tally_chains(self, chains, nodes, connections=None):
        """
        Tally chain types and return a dictonary with counts of the types.

        Parameters
        ----------
        chains : list
            All of the chains in the molecule

        nodes : list
            All of the element labels of the atoms

        connections : dict, key->list of keys
            A dictonary edge table with all the bidirectional connections

        Returns
        -------
        results : dict, labelled_chain->int
            Totals of the number of each type of chain
        """
        results = {}
        for chain in chains:
            labelled = tuple(nodes[x] for x in chain)
            if self.use_coordination:
                extra = tuple(str(len(connections[x])) for x in chain)
                labelled = tuple(x + y for x, y in zip(labelled, extra))

            if needs_reversal(labelled):
                labelled = labelled[::-1]
                chain = chain[::-1]
            labelled = self._convert_to_bond_order(chain, labelled,
                                                   connections)

            labelled = tuple(labelled)
            if labelled not in results:
                results[labelled] = 0
            results[labelled] += 1
        return results

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
        value : list
            All the chains in the molecule
        """
        data = self.convert_input(X)
        chains = self._loop_depth(data.connections)
        all_counts = self._tally_chains(chains, data.elements,
                                        data.connections)
        return list(all_counts.keys())

    def _idf(self, all_keys):
        res = defaultdict(float)
        for mol in all_keys:
            for key in mol:
                res[key] += 1
        N = len(all_keys)
        return {key: numpy.log(N / x) for key, x in res.items()}

    def fit(self, X, y=None):
        res = self.map(self._para_fit, X)
        vals = self.reduce(lambda x, y: set(x) | set(y), res)
        self._base_chains = tuple(sorted(vals))

        if self.do_tfidf:
            self._idf_values = self._idf(res)
        return self

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
        value : list
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.
        """
        self.check_fit()

        data = self.convert_input(X)
        chains = self._loop_depth(data.connections)
        tallies = self._tally_chains(chains, data.elements, data.connections)

        vector = []
        for x in self._base_chains:
            value = tallies.get(x, 0)
            if self.do_tfidf:
                value *= self._idf_values[x]
            vector.append(value)

        if self.add_unknown:
            unknown = 0
            for key, value in tallies.items():
                if key not in self._base_chains:
                    unknown += value
            vector.append(unknown)
        return vector

    def get_chain_labels(self, chains):
        if self.use_bond_order:
            return ['_'.join(['-'.join(y) for y in x]) for x in chains]
        return ['-'.join(x) for x in chains]


class Autocorrelation(BaseFeature):
    """
    A molecular descriptor based on Autocorrelation functions for properties.

    This is a compact (only depends on the number of properties used and the
    number of depths) molecule representation that uses the graph distance
    between atoms to extract information.

    .. math::

        V_d = \sum_i \sum_j P_i P_j \delta(d_{ij}, d)


    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    depths : list/tuple, default=None
        A list of depths to use for computing the autocorrelations functions.
        If this value is None, [0, 1, 2, 3] will be used.

    properties : list/tuple, default=None
        A list/tuple of properties to use. Each of these properties should be
        defined for a single atom in the molecule. Each property can be either
        a function (that takes in a LazyValues function and returns a vector
        the with one element per atom) or it can be a one of the following
        strings ('Z', 'EN', 'CN', 'I', 'R'). Each of these keys corresponds
        to the atomic number, the electronegativity, coordination number, the
        identity function (always returns 1), and the covalent radius.
        If this value is None, then all the predefined properties will be
        used.

    References
    ----------
    Janet, J. P. and  Kulik, H. J. Resolving Transition Metal Chemical Space:
    Feature Selection for Machine Learning and Structure-Property
    Relationships. J. Phys. Chem. A 2017, 121, 8939-8954
    """
    ATTRIBUTES = None
    LABELS = ('_labels', )

    def __init__(self, input_type='list', n_jobs=1, depths=None,
                 properties=None):
        super(Autocorrelation, self).__init__(input_type=input_type,
                                              n_jobs=n_jobs)
        self.functions = {
            'Z': lambda data: data.numbers,
            'EN': lambda data: [ELECTRONEGATIVITY[x] for x in data.elements],
            'CN': lambda data: [len(value) for key, value in
                                data.connections.items()],
            'I': lambda data: [1 for x in data.numbers],
            'R': lambda data: [BOND_LENGTHS[x]['1'] for x in data.elements],
        }
        if depths is None:
            depths = list(range(4))
        self.depths = depths
        if properties is None:
            properties = self.functions.keys()
        self.properties = sorted(properties, key=lambda x: str(x))
        self._labels = ['%s_%s' % pair for pair in
                        product(self.properties, self.depths)]

    def fit(self, X, y=None):
        """No fitting is required because it is defined by the parameters."""
        return self

    def _para_transform(self, X):
        self.check_fit()

        data = self.convert_input(X)
        D = get_graph_distance(data.connections)

        res = []
        for prop in self.properties:
            if callable(prop):
                p = prop(data)
            else:
                p = self.functions[prop](data)

            P = numpy.outer(p, p)
            for d in self.depths:
                res.append(((D == d) * P).sum())
        return res


class EncodedAngle(SetMergeMixin, EncodedFeature):
    r"""
    A smoothed histogram of atomic angles.

    This method is similar to EncodedBond but for angles in molecules. This is
    done by enumerating all triplets of atoms and computing the angle between
    them. The bins are then smoothed with smoothing functions. Note: The
    angles used are 0 to \pi.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    segments : int, default=40
        The number of bins/segments to use when generating the histogram.
        Empirically, it has been found that there is no benefit to having more
        than 40-50 segments.

    smoothing : string or callable, default='norm'
        A string or callable to use to smooth the histogram values. If a
        callable is given, it must take just a single argument that is a float
        (or vector of floats). For a list of supported default functions look
        at SMOOTHING_FUNCTIONS.

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

    form : int, default=3
        The histogram splitting style to use. This changes the scaling of
        this method to be O(E^3), O(E^2), O(E), or O(1) for 3, 2, 1, or 0
        respectively (where E is the number of elements).

    r_cut : float, default=6.
        The maximum distance allowed for atoms to be considered local to the
        "central atom".

    add_unknown : boolean, default=False
        Specifies whether or not to include an extra UNKNOWN count in the
        feature vector.

    Attributes
    ----------
    _groups : tuple, tuples
        A tuple of all the groups (element chains) in the fit molecules.
    """
    ATTRIBUTES = ("_groups", )
    LABELS = (("get_encoded_labels", "_groups"), )

    def __init__(self, input_type='list', n_jobs=1, segments=40,
                 smoothing="norm", slope=20., min_depth=0, max_depth=0,
                 form=3, r_cut=6., add_unknown=False):
        super(EncodedAngle, self).__init__(input_type=input_type,
                                           n_jobs=n_jobs, segments=segments,
                                           smoothing=smoothing, slope=slope,
                                           start=0., end=numpy.pi)
        self._groups = None
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.form = form
        self.r_cut = r_cut
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
        value : list
            All the elements in the molecule
        """
        # The creation of these triples could be done in several different
        # ways. Other approaches include filtering "impossible" duplicates
        # such as C-H-C in methane, or creating only the all combinations of
        # element triples. Loosely, this is something like:
        # combinations(sum([[ele]*min(num) for
        #              ele, num in Counter(elements)], []), 3).
        # In all the tests that were done, these approaches, while they might
        # be more "correct", perform significantly worse than the procedure
        # here (in terms of predictions). Similar issues arise in
        # _para_transform if i == k is removed.

        data = self.convert_input(X)
        pairs = get_element_pairs(data.elements)
        res = []
        for pair1 in pairs:
            for pair2 in pairs:

                for i in (0, 1):
                    # select the other index to use
                    inv_i = int(not i)
                    for j in (0, 1):
                        if pair1[i] != pair2[j]:
                            continue
                        inv_j = int(not j)
                        temp = (pair2[inv_j], pair1[i], pair1[inv_i])
                        res.append(temp)
        return set([x if x[0] < x[2] else x[::-1] for x in res])

    def f_c(self, R):
        return cosine_decay(R, r_cut=self.r_cut)

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
                    F = f_c[i, j] * f_c[j, k] * f_c[i, k]
                    eles = ele1, ele2, ele3
                    try:
                        idx = (get_index(eles), )
                    except KeyError:
                        idx = None
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
        value : list
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.
        """
        self.check_fit()
        data = self.convert_input(X)
        get_index, length, both = get_index_mapping(self._groups,
                                                    self.form,
                                                    self.add_unknown)
        iterator = self._iterator(data, get_index, both)
        return self.encode_values(iterator, (length, ))


class EncodedBond(SetMergeMixin, EncodedFeature):
    """
    A smoothed histogram of atomic distances.

    This is a method to generallize the idea of bond counting. Instead of
    seeing bonds as a discrete count that is thresholded at a given length,
    they are seen as general distance histograms. This is supplemented with
    smoothing functions.

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
        Empirically, it has been found that values beyond 50-100 have little
        benefit.

    smoothing : string or callable, default='norm'
        A string or callable to use to smooth the histogram values. If a
        callable is given, it must take just a single argument that is a float
        (or vector of floats). For a list of supported default functions look
        at SMOOTHING_FUNCTIONS.

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

    form : int, default=2
        The histogram splitting style to use. This changes the scaling of this
        method to be O(E^2), O(E), or O(1) for 2, 1, or 0 respectively (where
        E is the number of elements).

    add_unknown : boolean, default=False
        Specifies whether or not to include an extra UNKNOWN count in the
        feature vector.

    Attributes
    ----------
    _element_pairs : tuple, tuples
        A tuple of all the element pairs in the fit molecules.

    References
    ----------
    Collins, C.; Gordon, G.; von Lilienfeld, O. A.; Yaron, D. Constant Size
    Molecular Descriptors For Use With Machine Learning. arXiv:1701.06649
    """
    ATTRIBUTES = ("_element_pairs", )
    LABELS = (("get_encoded_labels", "_element_pairs"), )

    def __init__(self, input_type='list', n_jobs=1, segments=100,
                 smoothing='norm', start=0.2, end=6.0, slope=20.,
                 min_depth=0, max_depth=0, spacing='linear', form=2,
                 add_unknown=False):
        super(EncodedBond, self).__init__(input_type=input_type,
                                          n_jobs=n_jobs, segments=segments,
                                          smoothing=smoothing, start=start,
                                          end=end, slope=slope,
                                          spacing=spacing)
        self._element_pairs = None
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
        value : list
            All the element pairs in the molecule
        """
        data = self.convert_input(X)
        return get_element_pairs(data.elements)

    def _iterator(self, data, get_index, both):
        mat = get_depth_threshold_mask_connections(data.connections,
                                                   max_depth=self.max_depth,
                                                   min_depth=self.min_depth)
        distances = cdist(data.coords, data.coords)
        for i, ele1 in enumerate(data.elements):
            for j, ele2 in enumerate(data.elements):
                if i > j and not both:
                    continue
                if i == j or not mat[i, j]:
                    continue
                eles = (ele1, ele2)
                try:
                    idx = (get_index(eles), )
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
        value : list
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.
        """
        self.check_fit()
        data = self.convert_input(X)
        get_index, length, both = get_index_mapping(self._element_pairs,
                                                    self.form,
                                                    self.add_unknown)
        iterator = self._iterator(data, get_index, both)
        return self.encode_values(iterator, (length, ))


class CoulombMatrix(BaseFeature):
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

    drop_values : bool, default=False
        Specifies whether or not to drop the atoms from molecules larger than
        the training set. If this value is set to False, and the molecule is
        too large to transform, the transform will throw a ValueError. If it is
        set to True, then it will truncate the molecule to only include the
        first _max_size atoms of the molecule.

    Attributes
    ----------
    _max_size : int
        The size of the largest molecule in the fit molecules by number of
        atoms.

    References
    ----------
    Rupp, M.; Tkatchenko, A.; Muller, K.-R.; von Lilienfeld, O. A. Fast and
    Accurate Modeling of Molecular Atomization Energies with Machine Learning.
    Phys. Rev. Lett. 2012, 108, 058301.

    Hansen, K.; Montavon, G.; Biegler, F.; Fazli, S.; Rupp, M.; Scheffler, M.;
    von Lilienfeld, O. A.; Tkatchenko, A.; Muller, K.-R. Assessment and
    Validation of Machine Learning Methods for Predicting Molecular
    Atomization Energies. J. Chem. Theory Comput. 2013, 9, 3404-3419.
    """
    ATTRIBUTES = ("_max_size", )
    LABELS = (("get_coulomb_labels", "_max_size"), )

    def __init__(self, input_type='list', n_jobs=1, sort=False, eigen=False,
                 drop_values=False):
        super(CoulombMatrix, self).__init__(input_type=input_type,
                                            n_jobs=n_jobs)
        self._max_size = None
        self.sort = sort
        self.eigen = eigen
        self.drop_values = drop_values

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
        value : int
            The number of atoms in the molecule
        """
        data = self.convert_input(X)
        return len(data.elements)

    def fit(self, X, y=None):
        """
        Fit the model.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        max_size = self.map(self._para_fit, X)
        self._max_size = max(max_size)
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
            if not self.drop_values:
                msg = "The fit molecules (%d) were not as large as the ones"
                msg += " that are being transformed (%d)."
                raise ValueError(msg % (self._max_size, len(data.numbers)))
            numbers = data.numbers[:self._max_size]
            coords = data.coords[:self._max_size, :]
        else:
            numbers = data.numbers
            coords = data.coords

        padding_difference = self._max_size - len(numbers)
        values = get_coulomb_matrix(numbers, coords)
        if self.sort:
            order = numpy.argsort(values.sum(0))[::-1]
            values = values[order, :][:, order]

        if self.eigen:
            values = numpy.linalg.eig(values)[0]

        values = numpy.pad(values,
                           (0, padding_difference),
                           mode="constant")
        return values.reshape(-1)

    def get_coulomb_labels(self, max_size):
        if self.eigen:
            return ['coul-%d' % i for i in range(max_size)]
        labels = []
        for i in range(max_size):
            for j in range(max_size):
                labels.append('coul-%d-%d' % (i, j))
        return labels


class BagOfBonds(BaseFeature):
    """
    A molecular descriptor that groups interactions from the Coulomb Matrix.

    This feature starts the same as the Coulomb Matrix, and then interaction
    terms of the same element pair are grouped together and then sorted before
    they are vectorized.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    drop_values : bool, default=False
        Specifies whether or not to drop interactions if there are more than
        was seen in the training set. If this value is set to False, and the
        molecule is too large to transform, it will throw a ValueError. If it
        is set to True, then it will truncate that particular bag to only
        include the largest _bag_sizes[ele1, ele2] of the molecule.

    Attributes
    ----------
    _bag_sizes : dict, element pair->int
        A dictonary mapping element pairs to the maximum size of that element
        pair block in all the fit molecules.

    References
    ----------
    Hansen, K.; Biegler, F.; Ramakrishnan, R.; Pronobis, W.; von Lilienfeld,
    O. A.; Muller, K.-R.; Tkatchenko, A. Machine Learning Predictions of
    Molecular Properties: Accurate Many-body Potentials and Nonlocality in
    Chemical Space. J. Phys. Chem. Lett. 2015, 6, 2326-2331.
    """
    ATTRIBUTES = ("_bag_sizes", )
    LABELS = (("get_bob_labels", "_bag_sizes"), )

    def __init__(self, input_type='list', n_jobs=1, drop_values=False):
        super(BagOfBonds, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self._bag_sizes = None
        self.drop_values = drop_values

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
        value : list
            All the element pairs in the molecule
        """
        data = self.convert_input(X)
        bags = {}

        local = {}
        for element in data.elements:
            if element not in local:
                local[element] = 0
            local[element] += 1

        for i, ele1 in enumerate(local.keys()):
            for j, ele2 in enumerate(local.keys()):
                if j > i:
                    continue
                if ele1 == ele2:
                    # Minus 1 is to remove the diagonal
                    num = local[ele1] - 1
                    # Using Gauss summation trick
                    new_value = num * (num + 1) // 2
                else:
                    new_value = local[ele1] * local[ele2]

                sorted_ele = tuple(sorted([ele1, ele2]))
                bags[sorted_ele] = max(new_value, bags.get(sorted_ele, 0))
        return {key: value for key, value in bags.items() if value}

    def _max_merge_dict(self, x, y):
        """
        Merge the values of two dictonaries using the max of their values.

        Parameters
        ----------
        x : dict, key->number

        y : dict, key->number

        Returns
        -------
        dict : dict, key->number
        """
        all_keys = tuple(x.keys()) + tuple(y.keys())
        return {key: max(x.get(key, 0), y.get(key, 0)) for key in all_keys}

    def fit(self, X, y=None):
        """
        Fit the model.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        bags = self.map(self._para_fit, X)
        d = self.reduce(self._max_merge_dict, bags)
        self._bag_sizes = tuple(sorted(d.items()))
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

        Raises
        ------
        ValueError
            If the transformer has not been fit.

        ValueError
            If the size of the transforming molecules have more values in at
            least one bag than the same bag from the are larger than the fit.
        """
        self.check_fit()

        data = self.convert_input(X)
        # Sort the elements and coords based on the element
        temp = sorted(zip(data.elements, data.coords), key=lambda x: x[0])
        elements, coords = zip(*temp)

        bags = {k: [0 for i in range(v)] for k, v in self._bag_sizes}
        coulomb_matrix = get_coulomb_matrix(data.numbers, coords)

        ele_array = numpy.array(elements)
        for ele1, ele2 in bags.keys():
            # Select only the rows that are of type ele1
            first = ele_array == ele1

            # Select only the cols that are of type ele2
            second = ele_array == ele2
            # Select only the rows/cols that are in the upper triangle
            # (This could also be the lower), and are in a row, col with
            # ele1 and ele2 respectively
            mask = numpy.triu(numpy.logical_and.outer(first, second), k=1)
            # Add to correct double element bag highest to lowest
            values = sorted(coulomb_matrix[mask].tolist(), reverse=True)

            # The molecule being used was fit to something smaller
            if len(values) > len(bags[ele1, ele2]):
                if not self.drop_values:
                    msg = "The size of the %s bag is too small for this input"
                    raise ValueError(msg % ((ele1, ele2), ))
                values = values[:len(bags[ele1, ele2])]

            bags[ele1, ele2][:len(values)] = values
        order = [x[0] for x in self._bag_sizes]
        return sum((bags[key] for key in order), [])

    def get_bob_labels(self, bag_sizes):
        labels = []
        for bag, size in bag_sizes:
            name = '-'.join(bag)
            labels.extend(['%s_%d' % (name, i) for i in range(size)])
        return labels
