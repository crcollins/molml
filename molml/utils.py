import numpy
from scipy.spatial.distance import cdist
from scipy.special import expit
import scipy.stats


ELE_TO_NUM = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
}
NUM_TO_ELE = {y: x for x, y in ELE_TO_NUM.items()}


TYPE_ORDER = ['1', 'Ar', '2', '3']
BOND_LENGTHS = {
    "C": {
        "3":   0.62,
        "2":   0.69,
        "Ar": 0.72,
        "1":   0.85,
    },
    "Cl": {
        "1":   1.045,
    },
    "F": {
        "1":   1.23,
    },
    "H": {
        "1":   0.6,
    },
    "N": {
        "3":   0.565,
        "2":   0.63,
        "Ar": 0.655,
        "1":   0.74,
    },
    "O": {
        "3":   0.53,
        "2":   0.59,
        "Ar": 0.62,
        "1":   0.695,
    },
    "P": {
        "2":   0.945,
        "Ar": 0.985,
        "1":   1.11,
    },
    "S": {
        "2":   0.905,
        "Ar": 0.945,
        "1":   1.07,
    },
}


SMOOTHING_FUNCTIONS = {
    "norm_cdf": scipy.stats.norm.cdf,
    "zero_one": lambda x: (x > 0.).astype(float),
    "expit": expit,
    "tanh": lambda x: (numpy.tanh(x)+1) / 2,
    "norm": scipy.stats.norm.pdf,
    "expit_pdf": scipy.stats.logistic.pdf,
    "spike": lambda x: (numpy.abs(x) < 1.).astype(float)
}


def get_bond_type(element1, element2, dist):
    '''
    Given a pair of elements and a distance, return the bond order between
    them. If there is no bond, return None.
    '''
    for key in TYPE_ORDER[::-1]:
        try:
            cutoff = BOND_LENGTHS[element1][key] + BOND_LENGTHS[element2][key]
            if dist < cutoff:
                return key
        except KeyError:
            continue


def get_connections(elements, coords):
    '''
    Returns a dictonary edge list. Each value is is a tuple of the index of
    the connecting atom and the bond order as a string. Where the bond order
    is one of ['1', 'Ar', '2', '3'].

    Example HCN
    {
        0: {1: '1', 2: '3'},
        1: {0: '1'},
        2: {0: '3'},
    }
    '''
    dist_mat = cdist(coords, coords)

    connections = {i: {} for i in xrange(len(elements))}
    for i, element1 in enumerate(elements):
        for j, element2 in enumerate(elements[i+1:]):
            j += i + 1
            dist = dist_mat[i, j]
            bond_type = get_bond_type(element1, element2, dist)
            if not bond_type:
                continue

            # Loop over both connection directions
            # A -> B and A <- B
            for x, y in ((i, j), (j, i)):
                connections[x][y] = bond_type
    return connections


def read_file_data(path):
    """
    Reads a file and extracts the molecules geometry

    The file should be in the format
    ele0 x0 y0 z0
    ele1 x1 y1 z1
    ...
    """
    elements = []
    numbers = []
    coords = []
    with open(path, 'r') as f:
        for line in f:
            ele, x, y, z = line.strip().split()
            point = (float(x), float(y), float(z))
            elements.append(ele)
            numbers.append(ELE_TO_NUM[ele])
            coords.append(point)
    return elements, numbers, numpy.array(coords)


def get_depth_threshold_mask_connections(connections, max_depth=1):
    mat = numpy.zeros((len(connections), len(connections)))
    for key, values in connections.items():
        for val in values:
            mat[key, val] = 1
    return get_depth_threshold_mask(mat, max_depth)


def get_depth_threshold_mask(mat, max_depth=1):
    '''
    Given a connectivity matrix (either strings or ints), return a mask that is
    True at [i,j] if there exists a path from i to j that is of length
    `max_depth` or fewer.
    This is done by repeated matrix multiplication of the connectivity matrix.
    If `max_depth` is less than 1, this will return all True array.
    '''
    if max_depth < 1:
        temp = numpy.ones(mat.shape).astype(bool)
        return numpy.array(temp)

    mask = mat.copy().astype(bool)
    d = numpy.matrix(mat).astype(int)
    acc = d.copy()
    for i in xrange(2, max_depth + 1):
        acc *= d
        mask |= (acc == 1)
    return numpy.array(mask)


class LazyValues(object):
    '''
    An object to store molecule graph properties in a lazy fashion.

    This object allows only needing to compute different molecule graph
    properties if they are needed. The prime example of this being the
    computation of connections.

    Parameters
    ----------
    connections : dict, key->list of keys, default=None
        A dictonary edge table with all the bidirectional connections.

    numbers : array-like, shape=(n_atoms, ), default=None
        The atomic numbers of all the atoms.

    coords : array-like, shape=(n_atoms, 3), default=None
        The xyz coordinates of all the atoms (in angstroms).

    elements : array-like, shape=(n_atoms, ), default=None
        The element symbols of all the atoms.


    Attributes
    ----------
    connections : dict, key->list of keys
        A dictonary edge table with all the bidirectional connections. If the
        initialized value for this was None, then this will be computed from
        the coords and numbers/elements.

    numbers : array-like, shape=(n_atoms, )
        The atomic numbers of all the atoms. If the initialized value for this
        was None, then this will be computed from the elements.

    coords : array-like, shape=(n_atoms, 3)
        The xyz coordinates of all the atoms (in angstroms).

    elements : array-like, shape=(n_atoms, )
        The element symbols of all the atoms. If the initialized value for this
        was None, then this will be computed from the numbers.
    '''
    def __init__(self, connections=None, coords=None, numbers=None,
                 elements=None):
        self._connections = connections
        self._coords = coords
        self._numbers = numbers
        self._elements = elements

    @property
    def connections(self):
        if self._connections is None:
            self._connections = get_connections(self.elements, self.coords)
        return self._connections

    @property
    def coords(self):
        if self._coords is None:
            raise ValueError("No coordinates exist.")
        return self._coords

    @property
    def numbers(self):
        if self._numbers is None:
            if self._elements is not None:
                self._numbers = [ELE_TO_NUM[x] for x in self._elements]
            else:
                raise ValueError("No elements to convert to numbers.")
        return self._numbers

    @property
    def elements(self):
        if self._elements is None:
            if self._numbers is not None:
                self._elements = [NUM_TO_ELE[x] for x in self._numbers]
            else:
                raise ValueError("No numbers to convert to elements.")
        return self._elements
