import numpy
from scipy.spatial.distance import cdist

from base import BaseFeature
from utils import get_depth_threshold_mask_connections, get_coulomb_matrix
from utils import SPACING_FUNCTIONS, SMOOTHING_FUNCTIONS


class Shell(BaseFeature):
    '''
    A feature that counts the number of elements in a distance shell from the
    starting atom. This is similar to the features developed in Qu et. al.
    with the exception that it is atom-based rather than bond-based.

    References
    ----------
    Qu, X.; Latino, D. A.; Aires-de Sousa, J. A Big Data Approach to the
    Ultra-fast Prediction of DFT-calculated Bond Energies. J. Cheminf. 2013,
    5, 34.

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

    Attributes
    ----------
    _elements : set
        All the elements/types that are in the fit molecules.
    '''
    def __init__(self, input_type='list', n_jobs=1, depth=1,
                 use_coordination=False):
        super(Shell, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self.depth = depth
        self.use_coordination = use_coordination
        self._elements = None

    def _loop_depth(self, start, connections):
        '''
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
        '''
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
        '''
        Tally all the limit values and return a dictonary with all the counts
        of the types.

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
        '''
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
        value : set
            All the elements in the molecule
        '''
        data = self.convert_input(X)
        # This is just a cheap way to approximate the actual value
        elements = data.elements
        connections = data.connections
        if self.use_coordination:
            elements = [ele + str(len(connections[i])) for i, ele in
                        enumerate(elements)]
        return set(elements)

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
        results = self.map(self._para_fit, X)
        self._elements = set(self.reduce(lambda x, y: set(x) | set(y),
                                         results))
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
        value : list, shape=(n_atoms, len(self._elements))
            The features extracted from the molecule
        '''
        if self._elements is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        data = self.convert_input(X)

        vectors = []
        for atom in xrange(len(data.elements)):
            limits = self._loop_depth(atom, data.connections)
            tallies = self._tally_limits(limits, data.elements,
                                         data.connections)
            vectors.append([tallies.get(x, 0) for x in self._elements])
        return vectors


class LocalEncodedBond(BaseFeature):
    '''
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

    Attributes
    ----------
    _element : list
        A list of all the element pairs in the fit molecules.
    '''
    def __init__(self, input_type='list', n_jobs=1, segments=100,
                 smoothing="norm", start=0.2, end=6.0, slope=20., max_depth=0,
                 spacing="linear"):
        super(LocalEncodedBond, self).__init__(input_type=input_type,
                                               n_jobs=n_jobs)
        self._elements = None
        self.segments = segments
        self.smoothing = smoothing
        self.start = start
        self.end = end
        self.slope = slope
        self.max_depth = max_depth
        self.spacing = spacing

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
        value : set
            All the element pairs in the molecule
        '''
        data = self.convert_input(X)
        # This is just a cheap way to approximate the actual value
        return set(data.elements)

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
        self._elements = set(self.reduce(lambda x, y: set(x) | set(y),
                                         pairs))
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
        value : list, shape=(n_atoms, len(self._elements) * self.segments)
            The features extracted from the molecule
        '''
        if self._elements is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        try:
            smoothing_func = SMOOTHING_FUNCTIONS[self.smoothing]
        except KeyError:
            msg = "The value '%s' is not a valid spacing type."
            raise KeyError(msg % self.smoothing)

        pair_idxs = {key: i for i, key in enumerate(self._elements)}

        data = self.convert_input(X)

        vector = numpy.zeros((len(data.elements), len(self._elements),
                              self.segments))

        try:
            theta_func = SPACING_FUNCTIONS[self.spacing]
        except KeyError:
            msg = "The value '%s' is not a valid spacing type."
            raise KeyError(msg % self.spacing)

        theta = numpy.linspace(theta_func(self.start), theta_func(self.end),
                               self.segments)
        mat = get_depth_threshold_mask_connections(data.connections,
                                                   max_depth=self.max_depth)

        distances = cdist(data.coords, data.coords)
        for i, ele1 in enumerate(data.elements):
            for j, ele2 in enumerate(data.elements):
                if i == j or not mat[i, j]:
                    continue

                diff = theta - theta_func(distances[i, j])
                value = smoothing_func(self.slope * diff)
                vector[i, pair_idxs[ele2]] += value
        return vector.reshape(len(data.elements), -1)


class LocalCoulombMatrix(BaseFeature):
    '''
    An implementation of the Coulomb Matrix where only the local atom
    environment is used by using a cutoff radius.

    References
    ----------
    Barker, J.; Bulin, J.;  Hamaekers, J. LC-GAP: Localized Coulomb Descriptors
    for the Gaussian Approximation Potential. 2016

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

        M_{ij} = Z_{p_i} Z_{p_j} / (  ||  R_{p_1} - R_{p_i} ||_2
                                    + ||  R_{p_1} - R_{p_j} ||_2
                                    + ||  R_{p_i} - R_{p_j} ||_2 ) ** \alpha
        M_{ii} = 0.5 Z_{p_i} ** 2.4
    '''
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
        '''
        No fitting is needed because it is all defined by the parameters?
        '''
        pass

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
        dist = cdist(data.coords, data.coords)

        numbers = numpy.array(data.numbers)
        coords = numpy.array(data.coords)

        vectors = []
        for i in xrange(len(numbers)):
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
                vectors.append(mat[sorting[0]].tolist() + numpy.diag(mat)[1:])
            else:
                vectors.append(mat[sorting].flatten())
        return numpy.array(vectors)
