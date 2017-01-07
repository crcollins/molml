from builtins import range

import numpy
from scipy.spatial.distance import cdist

from .base import BaseFeature
from .utils import get_depth_threshold_mask_connections, get_coulomb_matrix
from .utils import ELE_TO_NUM, SMOOTHING_FUNCTIONS, SPACING_FUNCTIONS
from .utils import get_element_pairs, cosine_decay, needs_reversal


class Connectivity(BaseFeature):
    '''
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

    Attributes
    ----------
    _base_chains : set, tuples
        All the chains that are in the fit molecules.
    '''
    def __init__(self, input_type='list', n_jobs=1, depth=1,
                 use_bond_order=False, use_coordination=False,
                 add_unknown=False):
        super(Connectivity, self).__init__(input_type=input_type,
                                           n_jobs=n_jobs)
        self.depth = depth
        self.use_bond_order = use_bond_order
        self.use_coordination = use_coordination
        self.add_unknown = add_unknown
        self._base_chains = None

    def _loop_depth(self, connections):
        '''
        Loop over the depth number expanding chains

        Parameters
        ----------
        connections : dict, key->list of keys
            A dictonary edge table with all the bidirectional connections

        Returns
        -------
        chains : list
            A list of key tuples of all the chains in the molecule
        '''
        chains = [(x, ) for x in connections]
        for i in range(self.depth - 1):
            chains = self._expand_chains(chains, connections)
        return chains

    def _expand_chains(self, initial, connections):
        '''
        This uses the connectivity information to add one more atom to each
        chain.

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
        '''
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
        '''
        Converts a chain based on just elements into one that includes bond
        order.

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
        '''
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
        '''
        Tally all the chain types and return a dictonary with all the counts of
        the types.

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
        '''
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
        '''
        A single instance of the fit procedure

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
        '''
        data = self.convert_input(X)
        chains = self._loop_depth(data.connections)
        all_counts = self._tally_chains(chains, data.elements,
                                        data.connections)
        return list(all_counts.keys())

    def fit(self, X, y=None):
        '''
        Fit the model

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        '''
        base_chains = self.map(self._para_fit, X)
        self._base_chains = set(self.reduce(lambda x, y: set(x) | set(y),
                                            base_chains))
        return self

    def _para_transform(self, X, y=None):
        '''
        A single instance of the transform procedure

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
        '''
        if self._base_chains is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        data = self.convert_input(X)
        chains = self._loop_depth(data.connections)
        tallies = self._tally_chains(chains, data.elements, data.connections)

        vector = [tallies.get(x, 0) for x in sorted(self._base_chains)]
        if self.add_unknown:
            unknown = 0
            for key, value in tallies.items():
                if key not in self._base_chains:
                    unknown += value
            vector.append(unknown)
        return vector


class EncodedAngle(BaseFeature):
    '''
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
        callable is given, it must take just a single argument that is a float.
        For a list of supported default functions look at SMOOTHING_FUNCTIONS.

    slope : float, default=20.
        A parameter to tune the smoothing values. This is applied as a
        multiplication before calling the smoothing function.

    max_depth : int, default=0
        A parameter to set the maximum geodesic distance to include in the
        interactions. A value of 0 signifies that all interactions are
        included.

    spacing : string, default="linear"
        The histogram interval spacing type. Must be one of ("linear",
        "inverse", or "log"). Linear spacing is normal spacing. Inverse takes
        and evaluates the distances as 1/r and the start and end points are
        1/x. For log spacing, the distances are evaluated as numpy.log(r)
        and the start and end points are numpy.log(x).

    form : string, default="pair"
        The histogram splitting style to use. Must be one of ("pair",
        "element", or "single"). The "pair" option is the standard encoded
        bond. The "element" option will group elements of the same type
        together. "single" will only create one histogram. These options
        change the scaling of this method to be O(E^2), O(E), or O(1) for
        "pair", "element", and "single" respectively (where E is the number
        of elements).

    Attributes
    ----------
    _elements : list
        A list of all the elements in the fit molecules.
    '''
    def __init__(self, input_type='list', n_jobs=1, segments=100,
                 smoothing="norm", slope=20., max_depth=0,
                 r_cut=6., spacing="linear"):
        super(EncodedAngle, self).__init__(input_type=input_type,
                                           n_jobs=n_jobs)
        self._groups = None
        self.segments = segments
        self.smoothing = smoothing
        self.slope = slope
        self.max_depth = max_depth
        self.spacing = spacing
        self.r_cut = r_cut

    def _para_fit(self, X):
        '''
        A single instance of the fit procedure

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
        '''
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

    def fit(self, X, y=None):
        '''
        Fit the model

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        '''
        eles = self.map(self._para_fit, X)
        self._groups = set(self.reduce(lambda x, y: x | y, eles))
        return self

    def f_c(self, R):
        return cosine_decay(R, r_cut=self.r_cut)

    def _para_transform(self, X, y=None):
        '''
        A single instance of the transform procedure

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
        '''
        if self._groups is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        try:
            smoothing_func = SMOOTHING_FUNCTIONS[self.smoothing]
        except KeyError:
            msg = "The value '%s' is not a valid smoothing type."
            raise KeyError(msg % self.smoothing)

        try:
            theta_func = SPACING_FUNCTIONS[self.spacing]
        except KeyError:
            msg = "The value '%s' is not a valid spacing type."
            raise KeyError(msg % self.spacing)

        data = self.convert_input(X)

        ele_idx = {group: i for i, group in enumerate(self._groups)}
        length = len(ele_idx)
        vector = numpy.zeros((length, self.segments))

        theta = numpy.linspace(theta_func(0.), theta_func(numpy.pi),
                               self.segments)
        mat = get_depth_threshold_mask_connections(data.connections,
                                                   max_depth=self.max_depth)

        distances = cdist(data.coords, data.coords)
        f_c = self.f_c(distances)
        diffs = data.coords - data.coords[:, None]
        lengths = numpy.linalg.norm(diffs, axis=2)
        unit = diffs / lengths[:, :, None]
        inner = numpy.einsum('ijk,jmk->ijm', unit, unit)
        angles = numpy.arccos(numpy.clip(inner, -1., 1.))
        for i, ele1 in enumerate(data.elements):
            for j, ele2 in enumerate(data.elements):
                if i == j or not mat[i, j]:
                    continue
                for k, ele3 in enumerate(data.elements):
                    if j == k or not mat[j, k]:
                        continue
                    if i > k:
                        continue
                    F = f_c[i, j] * f_c[j, k] * f_c[i, k]
                    diff = theta - theta_func(angles[i, j, k])
                    value = smoothing_func(self.slope * diff)
                    if ele1 < ele3:
                        eles = ele1, ele2, ele3
                    else:
                        eles = ele3, ele2, ele1
                    vector[ele_idx[eles]] += value * F
        return vector.flatten().tolist()


class EncodedBond(BaseFeature):
    '''
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
        callable is given, it must take just a single argument that is a float.
        For a list of supported default functions look at SMOOTHING_FUNCTIONS.

    start : float, default=0.2
        The starting point for the histgram sampling in angstroms.

    end : float, default=6.0
        The ending point for the histogram sampling in angstroms.

    slope : float, default=20.
        A parameter to tune the smoothing values. This is applied as a
        multiplication before calling the smoothing function.

    max_depth : int, default=0
        A parameter to set the maximum geodesic distance to include in the
        interactions. A value of 0 signifies that all interactions are
        included.

    spacing : string, default="linear"
        The histogram interval spacing type. Must be one of ("linear",
        "inverse", or "log"). Linear spacing is normal spacing. Inverse takes
        and evaluates the distances as 1/r and the start and end points are
        1/x. For log spacing, the distances are evaluated as numpy.log(r)
        and the start and end points are numpy.log(x).

    form : string, default="pair"
        The histogram splitting style to use. Must be one of ("pair",
        "element", or "single"). The "pair" option is the standard encoded
        bond. The "element" option will group elements of the same type
        together. "single" will only create one histogram. These options
        change the scaling of this method to be O(E^2), O(E), or O(1) for
        "pair", "element", and "single" respectively (where E is the number
        of elements).

    Attributes
    ----------
    _element_pairs : list
        A list of all the element pairs in the fit molecules.
    '''
    def __init__(self, input_type='list', n_jobs=1, segments=100,
                 smoothing="norm", start=0.2, end=6.0, slope=20., max_depth=0,
                 spacing="linear", form="pair"):
        super(EncodedBond, self).__init__(input_type=input_type,
                                          n_jobs=n_jobs)
        self._element_pairs = None
        self.segments = segments
        self.smoothing = smoothing
        self.start = start
        self.end = end
        self.slope = slope
        self.max_depth = max_depth
        self.spacing = spacing
        self.form = form

    def _para_fit(self, X):
        '''
        A single instance of the fit procedure

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
        '''
        data = self.convert_input(X)
        return get_element_pairs(data.elements)

    def fit(self, X, y=None):
        '''
        Fit the model

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        '''
        pairs = self.map(self._para_fit, X)
        self._element_pairs = set(self.reduce(lambda x, y: set(x) | set(y),
                                              pairs))
        return self

    def _get_index_mapping(self):
        funcs = {
            "pair": lambda key: pair_idxs[tuple(sorted(key))],
            "element": lambda key: element_idxs[key[0]],
            "single": lambda _: 0,
        }
        if self.form == 'pair':
            pairs = sorted(self._element_pairs)
            length = len(pairs)
            pair_idxs = {key: i for i, key in enumerate(pairs)}
        elif self.form == 'element':
            elements = sorted(set(sum(self._element_pairs, tuple())))
            length = len(elements)
            element_idxs = {key: i for i, key in enumerate(elements)}
        elif self.form == 'single':
            length = 1
        else:
            raise ValueError("'%s' is not a valid form type" % self.form)
        return funcs[self.form], length

    def _para_transform(self, X, y=None):
        '''
        A single instance of the transform procedure

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
        '''
        if self._element_pairs is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        try:
            smoothing_func = SMOOTHING_FUNCTIONS[self.smoothing]
        except KeyError:
            msg = "The value '%s' is not a valid smoothing type."
            raise KeyError(msg % self.smoothing)

        try:
            theta_func = SPACING_FUNCTIONS[self.spacing]
        except KeyError:
            msg = "The value '%s' is not a valid spacing type."
            raise KeyError(msg % self.spacing)

        get_index, length = self._get_index_mapping()
        data = self.convert_input(X)

        vector = numpy.zeros((length, self.segments))

        theta = numpy.linspace(theta_func(self.start), theta_func(self.end),
                               self.segments)
        mat = get_depth_threshold_mask_connections(data.connections,
                                                   max_depth=self.max_depth)

        distances = cdist(data.coords, data.coords)
        for i, ele1 in enumerate(data.elements):
            for j, ele2 in enumerate(data.elements):
                if i > j and self.form != 'element':
                    continue
                if i == j or not mat[i, j]:
                    continue

                diff = theta - theta_func(distances[i, j])
                value = smoothing_func(self.slope * diff)
                eles = (ele1, ele2)
                vector[get_index(eles)] += value
        return vector.flatten().tolist()


class CoulombMatrix(BaseFeature):
    '''
    A molecular descriptor based on Coulomb interactions.

    This is a feature that uses a Coulomb-like interaction between all atoms
    in the molecule to generate a matrix that is then vectorized.

    C_ij = Z_i Z_j / | r_i - r_j |
    C_ii = 0.5 Z_i ** 2.4

    References
    ----------
    Rupp, M.; Tkatchenko, A.; Muller, K.-R.; von Lilienfeld, O. A. Fast and
    Accurate Modeling of Molecular Atomization Energies with Machine Learning.
    Phys. Rev. Lett. 2012, 108, 058301.

    Hansen, K.; Montavon, G.; Biegler, F.; Fazli, S.; Rupp, M.; Scheffler, M.;
    von Lilienfeld, O. A.; Tkatchenko, A.; Muller, K.-R. Assessment and
    Validation of Machine Learning Methods for Predicting Molecular
    Atomization Energies. J. Chem. Theory Comput. 2013, 9, 3404-3419.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    Attributes
    ----------
    _max_size : int
        The size of the largest molecule in the fit molecules by number of
        atoms.
    '''
    def __init__(self, input_type='list', n_jobs=1):
        super(CoulombMatrix, self).__init__(input_type=input_type,
                                            n_jobs=n_jobs)
        self._max_size = None

    def _para_fit(self, X):
        '''
        A single instance of the fit procedure

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
        '''
        data = self.convert_input(X)
        return len(data.elements)

    def fit(self, X, y=None):
        '''
        Fit the model

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        '''
        max_size = self.map(self._para_fit, X)
        self._max_size = max(max_size)
        return self

    def _para_transform(self, X):
        '''
        A single instance of the transform procedure

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
        '''
        data = self.convert_input(X)
        if self._max_size is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)
        if len(data.numbers) > self._max_size:
            msg = "The fit molecules (%d) were not as large as the ones that"
            msg += " are being transformed (%d)."
            raise ValueError(msg % (self._max_size, len(data.numbers)))

        padding_difference = self._max_size - len(data.numbers)
        coulomb_matrix = get_coulomb_matrix(data.numbers, data.coords)
        new_coulomb_matrix = numpy.pad(coulomb_matrix,
                                       (0, padding_difference),
                                       mode="constant")
        return new_coulomb_matrix.reshape(-1)


class BagOfBonds(BaseFeature):
    '''
    A molecular descriptor that groups interactions from the Coulomb Matrix

    This feature starts the same as the Coulomb Matrix, and then interaction
    terms of the same element pair are grouped together and then sorted before
    they are vectorized.

    References
    ----------
    Hansen, K.; Biegler, F.; Ramakrishnan, R.; Pronobis, W.; von Lilienfeld,
    O. A.; Muller, K.-R.; Tkatchenko, A. Machine Learning Predictions of
    Molecular Properties: Accurate Many-body Potentials and Nonlocality in
    Chemical Space. J. Phys. Chem. Lett. 2015, 6, 2326-2331.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    Attributes
    ----------
    _bag_sizes : dict, element pair->int
        A dictonary mapping element pairs to the maximum size of that element
        pair block in all the fit molecules.
    '''
    def __init__(self, input_type='list', n_jobs=1):
        super(BagOfBonds, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self._bag_sizes = None

    def _para_fit(self, X):
        '''
        A single instance of the fit procedure

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
        '''
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
        '''
        Merge the values of two dictonaries using the max of their values

        Parameters
        ----------
        x : dict, key->number

        y : dict, key->number

        Returns
        -------
        dict : dict, key->number
        '''
        all_keys = tuple(x.keys()) + tuple(y.keys())
        return {key: max(x.get(key, 0), y.get(key, 0)) for key in all_keys}

    def fit(self, X, y=None):
        '''
        Fit the model

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        '''
        bags = self.map(self._para_fit, X)
        self._bag_sizes = self.reduce(self._max_merge_dict, bags)
        return self

    def _para_transform(self, X):
        '''
        A single instance of the transform procedure

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
        '''
        data = self.convert_input(X)
        if self._bag_sizes is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        # Sort the elements and coords based on the element
        temp = sorted(zip(data.elements, data.coords), key=lambda x: x[0])
        elements, coords = zip(*temp)

        bags = {key: [0 for i in range(value)]
                for key, value in self._bag_sizes.items()}
        numbers = [ELE_TO_NUM[x] for x in elements]
        coulomb_matrix = get_coulomb_matrix(numbers, coords)

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
                msg = "The size of the %s bag is too small for this input"
                raise ValueError(msg % ((ele1, ele2), ))

            bags[ele1, ele2][:len(values)] = values
        order = sorted(bags)
        return sum((bags[key] for key in order), [])
