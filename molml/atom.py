import numpy
from scipy.spatial.distance import cdist

from features import BaseFeature
from utils import get_depth_threshold_mask_connections, SMOOTHING_FUNCTIONS


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
    _elements : set, tuples
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
        value : list
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
        value : list
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

    Attributes
    ----------
    _element_pairs : list
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
        value : list
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
        value : list
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

        thetas = {
            "log": lambda x: numpy.log(x),
            "inverse": lambda x: 1 / x,
            "linear": lambda x: x,
        }
        try:
            theta_func = thetas[self.spacing]
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
