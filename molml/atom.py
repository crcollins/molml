from features import BaseFeature


class Shell(BaseFeature):
    '''

    '''
    def __init__(self, input_type='list', n_jobs=1, depth=1):
        super(Shell, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self.depth = depth
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

    def _tally_chains(self, limit, elements):
        counts = {}
        for x in limit:
            ele = elements[x]
            if ele not in counts:
                counts[ele] = 0
            counts[ele] += 1
        return counts

    def _para_fit(self, X):
        data = self.convert_input(X)
        # This is just a cheap way to approximate the actual value
        return set(data.elements)

    def fit(self, X, y=None):
        results = self.map(self._para_fit, X)
        self._elements = set(self.reduce(lambda x, y: set(x) | set(y),
                                         results))
        return self

    def _para_transform(self, X):
        data = self.convert_input(X)

        vectors = []
        for atom in xrange(len(data.elements)):
            limits = self._loop_depth(atom, data.connections)
            tallies = self._tally_chains(limits, data.elements)
            vectors.append([tallies.get(x, 0) for x in self._elements])
        return vectors
