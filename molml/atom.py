from features import BaseFeature


class Shell(BaseFeature):
    '''

    '''
    def __init__(self, input_type='list', n_jobs=1, depth=1,
                 use_coordination=False):
        super(Shell, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self.depth = depth
        self.use_coordination = use_coordination
        self._elements = None

    def _loop_depth(self, start, connections):
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

    def _tally_chains(self, limits, elements, connections=None):
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
        data = self.convert_input(X)
        # This is just a cheap way to approximate the actual value
        elements = data.elements
        connections = data.connections
        if self.use_coordination:
            elements = [ele + str(len(connections[i])) for i, ele in
                        enumerate(elements)]
        return set(elements)

    def fit(self, X, y=None):
        results = self.map(self._para_fit, X)
        self._elements = set(self.reduce(lambda x, y: set(x) | set(y),
                                         results))
        return self

    def _para_transform(self, X):
        if self._elements is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        data = self.convert_input(X)

        vectors = []
        for atom in xrange(len(data.elements)):
            limits = self._loop_depth(atom, data.connections)
            tallies = self._tally_chains(limits, data.elements,
                                         data.connections)
            vectors.append([tallies.get(x, 0) for x in self._elements])
        return vectors
