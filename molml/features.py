import multiprocessing

import numpy
from scipy.spatial.distance import cdist
from pathos.multiprocessing import ProcessingPool as Pool

from utils import get_connections, read_file_data
from utils import ELE_TO_NUM, SMOOTHING_FUNCTIONS


def _func_star(args):
    '''
    A function and argument expanding helper function

    The first item in args is callable, and the remainder of the items are
    used as expanded arguments. This is to make the function header for reduce
    the same for the normal and parallel versions. Otherwise, the functions
    would have to do their own unpacking of arguments.
    '''
    f = args[0]
    args = args[1:]
    return f(*args)


class BaseFeature(object):
    '''
    A base class for all the features.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.
    '''
    def __init__(self, input_type='list', n_jobs=1):
        self.input_type = input_type
        self.n_jobs = n_jobs

    def __repr__(self):
        return "%s(input_type='%s', n_jobs=%d)" % (
            type(self).__name__, self.input_type, self.n_jobs)

    def convert_input(self, X):
        '''
        Converts the input (as specified in self.input_type) to a usable form

        Parameters
        ----------
        X : list or string (depends on the instance value of input_type)
            If input_type is 'list', then it must be an iterable of (elements,
            coodinates pairs) for each molecule. Where the elements are an
            iterable of the form (ele1, ele2, ..., elen) and coordinates are an
            iterable of the form [(x1, y1, z1), (x2, y2, z2), ..., (xn, yn,
            zn)].

            If input_type is 'filename', then it must be an iterable of
            paths/filenames for each molecule. The files must then be of the 
            form
            ele1 x1 y1 z1
            ele2 x2 y2 z2
            ...
            elen xn yn zn

        Returns
        -------
        elements : list, shape=[n_atoms]
            A list of all the element symbols in the molecule

        coordinates : array, shape=[n_atoms, 3]
            An array of all the coordinates of the atoms in the molecule.
            These are assumed to be in angstroms.
        '''
        if self.input_type == "list":
            elements, coordinates = X
        elif self.input_type == "filename":
            elements, numbers, coordinates = read_file_data(X)
        else:
            raise ValueError("The input_type '%s' is not allowed." %
                                self.input_type)
        return elements, coordinates

    def map(self, f, seq):
        '''
        Parallel implementation of map

        Parameters
        ----------
        f : callable
            A function to map to all the values in 'seq'

        seq : iterable
            An iterable of values to process with 'f'

        Returns
        -------
        results : list, shape=[len(seq)]
            The evaluated values
        '''
        if self.n_jobs < 1:
            n_jobs = multiprocessing.cpu_count()
        elif self.n_jobs == 1:
            return map(f, seq)
        else:
            n_jobs = self.n_jobs

        pool = Pool(processes=n_jobs)
        results = pool.map(f, seq)
        return results

    def reduce(self, f, seq):
        '''
        Parallel implementation of reduce

        This changes the problem from being O(n) steps to O(lg n)

        Parameters
        ----------
        f : callable
            A function to use to reduce the values of 'seq'

        seq : iterable
            An iterable of values to process

        Returns
        -------
        results : object
            A single reduced object based on 'seq' and 'f'
        '''
        if self.n_jobs == 1:
            return reduce(f, seq)

        while len(seq) > 1:
            pairs = [(f, seq[i], seq[i + 1]) for i in xrange(0, len(seq) - 1, 2)]
            seq = self.map(_func_star, pairs) + [seq[-1]] * (len(seq) % 2)
        return seq[0]

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
        raise NotImplementedError

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
        value : array-like
            The features extracted from the molecule
        '''
        raise NotImplementedError

    def transform(self, X, y=None):
        '''
        Framework for a potentially parallel transform

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to transform

        Returns
        -------
        array : array, shape=(n_samples, n_features)
            The transformed features
        '''
        results = self.map(self._para_transform, X)
        return numpy.array(results)

    def fit_transform(self, X, y=None):
        '''
        A naive default implementation of fitting and transforming

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit and then transform

        Returns
        -------
        array : array, shape=(n_samples, n_features)
            The transformed features
        '''
        self.fit(X, y)
        return self.transform(X, y)


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
    '''
    def __init__(self, input_type='list', n_jobs=1, depth=1, use_bond_order=False):
        super(Connectivity, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self.depth = depth
        self.use_bond_order = use_bond_order
        self._base_chains = None

    def __repr__(self):
        return "%s(input_type='%s', n_jobs=%d, depth=%d, use_bond_order=%s)" % (
                    type(self).__name__, self.input_type, self.n_jobs,
                    self.depth, self.use_bond_order)

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
        chains = [(x, ) for x in connections.keys()]
        for i in xrange(self.depth - 1):
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
        return results.keys()

    def _get_ordering_idxs(self, x):
        '''
        This is used to select the two indicies that define the sorting order
        for the chains. The two returned values correspond to the lower and
        the higher values.

        Parameters
        ----------
        x : int
            An integer length of the chain

        Returns
        -------
        lower : int
            The lower index value when sorting

        upper : int
            The upper index value when sorting
        '''
        if x == 1:
            return 0, 0
        q, r = divmod(x, 2)
        return q - 1, q + r

    def _sort_chain(self, chain, labelled):
        '''
        Reorder chain

        Sort the chains such that they are in a canonical ordering

        Parameters
        ----------
        chain : tuple
            The atom index values in the chain

        labelled : tuple
            Elements corresponding to the chain indices

        Returns
        -------
        chain : tuple
            Atom index values for the sorted chain

        labelled : tuple
            Elements corresponding to the sorted chain indices
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

            chain, labelled = self._sort_chain(chain, labelled)
            labelled = self._convert_to_bond_order(chain, labelled, connections)

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
        elements, coords = self.convert_input(X)
        connections = get_connections(elements, coords)
        chains = self._loop_depth(connections)
        all_counts = self._tally_chains(chains, elements, connections)
        return all_counts.keys()

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
        f = lambda x, y: set(x) | set(y)
        self._base_chains = set(self.reduce(f, base_chains))
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

        elements, coords = self.convert_input(X)
        connections = get_connections(elements, coords)
        chains = self._loop_depth(connections)
        tallies = self._tally_chains(chains, elements, connections)

        return [tallies.get(x, 0) for x in self._base_chains]


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
    '''
    def __init__(self, input_type='list', n_jobs=1, segments=100, smoothing="norm", start=0.2, end=6.0, slope=20.):
        super(EncodedBond, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self._element_pairs = None
        self.segments = segments
        self.smoothing = smoothing
        self.start = start
        self.end = end
        self.slope = slope

    def __repr__(self):
        string = "%s(input_type='%s', n_jobs=%d, segments=%d, smoothing='%s', start=%g, end=%g, slope=%g)"
        return string % (type(self).__name__, self.input_type, self.n_jobs,
                        self.segments, self.smoothing, self.start, self.end, self.slope)

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
        elements, coords = self.convert_input(X)

        counts = {}
        for ele in elements:
            if ele not in counts:
                counts[ele] = 0
            counts[ele] += 1

        pairs = {}
        for i, x in enumerate(counts):
            for j, y in enumerate(counts):
                if i > j:
                    continue
                if x == y and counts[x] == 1:
                    continue
                if x > y:
                    pairs[y, x] = 1
                else:
                    pairs[x, y] = 1
        return pairs.keys()

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
        f = lambda x, y: set(x) | set(y)
        self._element_pairs = set(self.reduce(f, pairs))
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
        if self._element_pairs is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        smoothing_func = SMOOTHING_FUNCTIONS[self.smoothing]

        pair_idxs = {key: i for i, key in enumerate(self._element_pairs)}

        elements, coords = self.convert_input(X)
        vector = numpy.zeros((len(self._element_pairs), self.segments))

        theta = numpy.linspace(self.start, self.end, self.segments)

        distances = cdist(coords, coords)
        for i, ele1 in enumerate(elements):
            for j, ele2 in enumerate(elements[i + 1:]):
                j += i + 1
                value = smoothing_func(self.slope * (theta - distances[i, j]))
                if ele1 < ele2:
                    vector[pair_idxs[ele1, ele2]] += value
                else:
                    vector[pair_idxs[ele2, ele1]] += value
        return vector.flatten().tolist()


def get_coulomb_matrix(numbers, coords):
    """
    Return the coulomb matrix for the given coords and numbers

    C_ij = Z_i Z_j / | r_i - r_j |
    C_ii = 0.5 Z_i ** 2.4

    Parameters
    ----------
    numbers : array-like, shape=(n_atoms, )
        The atomic numbers of all the atoms

    coords : array-like, shape=(n_atoms, 3)
        The xyz coordinates of all the atoms (in angstroms)

    Returns
    -------
    top : array, shape=(n_atoms, n_atoms)
        The coulomb matrix
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
    '''
    A molecular descriptor based on Coulomb interactions.

    This is a feature that uses a Coulomb-like interaction between all atoms
    in the molecule to generate a matrix that is then vectorized.

    C_ij = Z_i Z_j / | r_i - r_j |
    C_ii = 0.5 Z_i ** 2.4

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.
    '''
    def __init__(self, input_type='list', n_jobs=1):
        super(CoulombMatrix, self).__init__(input_type=input_type, n_jobs=n_jobs)
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
        elements, coords = self.convert_input(X)
        return len(elements)

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
        elements, coords = self.convert_input(X)
        if self._max_size is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)
        if len(elements) > self._max_size:
            msg = "The fit molecules (%d) were not as large as the ones that"
            msg += " are being transformed (%d)."
            raise ValueError(msg % (self._max_size, len(elements)))

        padding_difference = self._max_size - len(elements)
        numbers = [ELE_TO_NUM[x] for x in elements]
        coulomb_matrix = get_coulomb_matrix(numbers, coords)
        new_coulomb_matrix = numpy.pad(coulomb_matrix, (0, padding_difference), mode="constant")
        return new_coulomb_matrix.reshape(-1)


class BagOfBonds(BaseFeature):
    '''
    A molecular descriptor that groups interactions from the Coulomb Matrix

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
        elements, coords = self.convert_input(X)
        bags = {}

        local = {}
        for element in elements:
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
                    new_value = num * (num + 1) / 2
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
        all_keys = x.keys() + y.keys()
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
        elements, coords = self.convert_input(X)
        if self._bag_sizes is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

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
                msg = "The size of the %s bag is too small for this input"
                raise ValueError(msg % ((ele1, ele2), ))

            bags[ele1, ele2][:len(values)] = values
        return sum(bags.values(), [])
