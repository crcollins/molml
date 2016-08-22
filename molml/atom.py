from features import BaseFeature


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
