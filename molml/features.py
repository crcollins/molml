import numpy
from scipy.spatial.distance import cdist
import scipy.stats


from utils import get_connections, ELE_TO_NUM


class BaseFeature(object):
    def __init__(self, input_type='list'):
        '''
        Currently, the only input_type supported is 'list'

        If the input is 'list', then it must be an iterable of (elements, coodinates pairs) for each molecule. 
        Where the elements are an iterable of the form (ele1, ele2, ..., elen) and coordinates are an iterable of the form 
        [(x1, y1, z1), (x2, y2, z2), ..., (xn, yn, zn)].
        '''
        self.input_type = input_type

    def __repr__(self):
        return "%s(input_type='%s')" % (self.__class__.__name__, self.input_type)

    def fit(self, X, y=None):
        raise NotImplementedError

    def _para_transform(self, X):
        '''A stupidly parallel implementation of the transformation'''
        raise NotImplementedError

    def transform(self, X, y=None):
        '''Framework for a potentially parallel transform'''
        results = map(self._para_transform, X)
        return numpy.array(results)

    def fit_transform(self, X, y=None):
        '''A naive default implementation of fitting and transforming'''
        self.fit(X, y)
        return self.transform(X, y)


class Connectivity(BaseFeature):
    '''
    A collection of feature types based on the connectivity of atoms.
    '''
    def __init__(self, input_type='list', depth=1, use_bond_order=False):
        super(Connectivity, self).__init__(input_type=input_type)
        self.depth = depth
        self.use_bond_order = use_bond_order
        self._base_chains = None

    def __repr__(self):
        return "%s(input_type='%s', depth='%d', use_bond_order=%s)" % (self.__class__.__name__, self.input_type, self.depth, self.use_bond_order)

    def _loop_depth(self, connections):
        '''
        Loop over the depth number expanding chains
        '''
        chains = [(x, ) for x in connections.keys()]
        for i in xrange(self.depth - 1):
            chains = self._expand_chains(chains, connections)
        return chains

    def _expand_chains(self, initial, connections):
        '''
        This uses the connectivity information to add one more atom to each 
        chain. Returns a list of index chains that are one index longer than
        the inputs in initial.
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
                if x in item: continue
                new = (x, ) + item
                if new[0] > new[-1]:
                    new = new[::-1]
                if new not in results:
                    results[new] = 1
        return results.keys()

    def _get_ordering_idxs(self, x):
        '''
        This is used to select the two indicies that define the sorting order
        for the chains. The two returned values correspond to the lower and
        the higher values. 

        Arguments:
            x: An integer length of the chain
        '''
        if x == 1:
            return 0, 0
        q, r = divmod(x, 2)
        return q - 1, q + r

    def _sort_chain(self, chain, labelled):
        '''
        Reorder chain

        Sort the chains such that they are in a canonical ordering

        '''
        first, second = self._get_ordering_idxs(len(labelled))
        while first >= 0 and second < len(labelled):
            if labelled[first] > labelled[second]:
                # Case where order reversal is needed
                labelled = labelled[::-1]
                chain = chain[::-1]
                break
            elif labelled[first] == labelled[second]:
                # Indeterminate case
                first -= 1
                second += 1
            else:
                # Case already in the correct order
                break
        return chain, labelled

    def _convert_to_bond_order(self, chain, labelled, connections):
        '''
        Converts a chain based on just elements into one that includes bond
        order.
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
        '''
        results = {}
        for chain in chains:
            labelled = tuple(nodes[x] for x in chain)

            chain, labelled = self._sort_chain(chain, labelled)
            labelled = self._convert_to_bond_order(chain, labelled, connections)

            labelled = tuple(labelled)
            if labelled not in results:
                results[labelled] = 0
            results[labelled] += 1
        return results

    def _para_fit(self, X):
        elements, coords = X
        connections = get_connections(elements, coords)
        chains = self._loop_depth(connections)
        all_counts = self._tally_chains(chains, elements, connections)
        return all_counts.keys()

    def fit(self, X, y=None):
        '''
        '''
        base_chains = map(self._para_fit, X)
        self._base_chains = reduce(lambda x, y: set(x) | set(y), base_chains)
        return self

    def _para_transform(self, X, y=None):
        '''
        '''
        if self._base_chains is None:
            raise ValueError

        elements, coords = X
        connections = get_connections(elements, coords)
        chains = self._loop_depth(connections)
        tallies = self._tally_chains(chains, elements, connections)

        return [tallies.get(x, 0) for x in self._base_chains]


class EncodedBond(BaseFeature):
    def __init__(self, input_type='list', segments=100, start=0.2, end=6.0, slope=20.):
        super(EncodedBond, self).__init__(input_type=input_type)
        self._element_pairs = None
        self.segments = segments
        self.start = start
        self.end = end
        self.slope = slope

    def __repr__(self):
        string = "%s(input_type='%s', segments=%d, start='%g', end='%g', slope='%g')"

        return string % (self.__class__.__name__, self.input_type, 
                        self.segments, self.start, self.end, self.slope)

    def _para_fit(self, X):
        elements, coords = X

        counts = {}
        for ele in elements:
            if ele not in counts:
                counts[ele] = 0
            counts[ele] += 1

        pairs = {}
        for i, x in enumerate(counts):
            for j, y in enumerate(counts):
                if i > j: continue
                if x == y and counts[x] == 1: continue
                if x > y:
                    pairs[y, x] = 1
                else:
                    pairs[x, y] = 1
        return pairs.keys()        

    def fit(self, X, y=None):
        '''
        '''
        pairs = map(self._para_fit, X)
        self._element_pairs = reduce(lambda x, y: set(x) | set(y), pairs)
        return self

    def _para_transform(self, X, y=None):
        if self._element_pairs is None:
            raise ValueError
            
        smoothing_function = scipy.stats.norm.pdf

        pair_idxs = {key: i for i, key in enumerate(self._element_pairs)}

        elements, coords = X
        vector = numpy.zeros((len(self._element_pairs), self.segments))

        theta = numpy.linspace(self.start, self.end, self.segments)
        theta = numpy.logspace(numpy.log(self.start), numpy.log(self.end), self.segments)
        theta = 1/numpy.linspace(1/self.start, 1/self.end, self.segments)

        distances = cdist(coords, coords)
        for i, ele1 in enumerate(elements):
            for j, ele2 in enumerate(elements[i + 1:]):
                j += i + 1
                value = smoothing_function(self.slope * (theta - distances[i, j]))
                if ele1 < ele2:
                    vector[pair_idxs[ele1, ele2]] += value
                else:
                    vector[pair_idxs[ele2, ele1]] += value
        return vector.flatten().tolist()


def get_coulomb_matrix(numbers, coords):
    """
    Return the coulomb matrix for the given `coords` and `numbers`
    """
    top = numpy.outer(numbers, numbers).astype(numpy.float64)
    r = cdist(coords, coords)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        numpy.divide(top, r, top)
    numpy.fill_diagonal(top, 0.5 * numpy.array(numbers) ** 2.4)
    top[top == numpy.Infinity] = 0
    top[numpy.isnan(top)] = 0
    return top


class CoulombMatrix(BaseFeature):
    def __init__(self, input_type='list'):
        super(CoulombMatrix, self).__init__(input_type=input_type)
        self._max_size = None

    def _para_fit(self, X):
        elements, coords = X
        return len(elements)

    def fit(self, X, y=None):
        max_size = map(self._para_fit, X)
        self._max_size = max(max_size)
        return self

    def _para_transform(self, X):
        elements, coords = X
        if self._max_size is None or len(elements) > self._max_size:
            raise ValueError

        padding_difference = self._max_size - len(elements)
        numbers = [ELE_TO_NUM[x] for x in elements]
        coulomb_matrix = get_coulomb_matrix(numbers, coords)
        new_coulomb_matrix = numpy.pad(coulomb_matrix, (0, padding_difference), mode="constant")
        return new_coulomb_matrix.reshape(-1)


class BagOfBonds(BaseFeature):
    def __init__(self, input_type='list'):
        super(BagOfBonds, self).__init__(input_type=input_type)
        self._bag_sizes = None

    def _para_fit(self, X):
        elements, coords = X
        bags = {}

        local = {}
        for element in elements:
            if element not in local:
                local[element] = 0
            local[element] += 1

        for i, ele1 in enumerate(local.keys()):
            for j, ele2 in enumerate(local.keys()):
                if j > i: continue
                if ele1 == ele2:
                    # Minus 1 is to remove the diagonal
                    num = local[ele1] - 1
                    # Using Gauss summation trick
                    new_value = num * (num + 1) / 2
                else:
                    new_value = local[ele1] * local[ele2]

                sorted_ele = tuple(sorted([ele1, ele2]))
                bags[sorted_ele] = max(new_value, bags.get(sorted_ele, 0))
        return {key: value for key, value in bags.items() if value}

    def _max_merge_dict(self, x, y):
        all_keys = x.keys() + y.keys()
        return {key: max(x.get(key, 0), y.get(key, 0)) for key in all_keys}

    def fit(self, X, y=None):
        bags = map(self._para_fit, X)
        self._bag_sizes = reduce(self._max_merge_dict, bags)
        return self

    def _para_transform(self, X):
        elements, coords = X
        if self._bag_sizes is None:
            raise ValueError

        # Sort the elements and coords based on the element
        temp = sorted(zip(elements, coords), key=lambda x: x[0])
        elements, coords = zip(*temp)

        bags = {key: [0 for i in xrange(value)] for key, value in self._bag_sizes.items()}
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
                raise ValueError

            bags[ele1, ele2][:len(values)] = values
        return sum(bags.values(), [])