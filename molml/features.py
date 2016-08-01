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
