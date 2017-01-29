"""
A collection of asssorted utility functions.
"""
from builtins import range

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
SPACING_FUNCTIONS = {
    "log": lambda x: numpy.log(x),
    "inverse": lambda x: 1 / x,
    "linear": lambda x: x,
}


def get_smoothing_function(key):
    try:
        return SMOOTHING_FUNCTIONS[key]
    except KeyError:
        msg = "The value '%s' is not a valid smoothing type."
        raise KeyError(msg % key)


def get_spacing_function(key):
    try:
        return SPACING_FUNCTIONS[key]
    except KeyError:
        msg = "The value '%s' is not a valid spacing type."
        raise KeyError(msg % key)


def get_bond_type(element1, element2, dist):
    """
    Get the bond type between two elements based on their distance.

    If there is no bond, return None.
    """
    for key in TYPE_ORDER[::-1]:
        try:
            cutoff = BOND_LENGTHS[element1][key] + BOND_LENGTHS[element2][key]
            if dist < cutoff:
                return key
        except KeyError:
            continue


def get_connections(elements, coords):
    """
    Return a dictonary edge list.

    Each value is is a tuple of the index of
    the connecting atom and the bond order as a string. Where the bond order
    is one of ['1', 'Ar', '2', '3'].

    Example HCN
    {
        0: {1: '1', 2: '3'},
        1: {0: '1'},
        2: {0: '3'},
    }
    """
    dist_mat = cdist(coords, coords)

    connections = {i: {} for i in range(len(elements))}
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
    Determine the file type and call the correct parser.

    The accepted file types are .out and .xyz files.
    """
    if path.endswith('.out'):
        return read_out_data(path)
    elif path.endswith('.xyz'):
        return read_xyz_data(path)
    else:
        raise ValueError("Unknown file type")


def read_out_data(path):
    """
    Read an out and extract the molecule's geometry.

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


def read_xyz_data(path):
    """
    Read an xyz file and extract the molecule's geometry.

    The file should be in the format
    num_atoms
    comment
    ele0 x0 y0 z0
    ele1 x1 y1 z1
    ...
    """
    elements = []
    numbers = []
    coords = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
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
    """
    Compute a depth threshold mask based on an adjacency matrix.

    Given a connectivity matrix (either strings or ints), return a mask that is
    True at [i,j] if there exists a path from i to j that is of length
    `max_depth` or fewer.
    This is done by repeated matrix multiplication of the connectivity matrix.
    If `max_depth` is less than 1, this will return all True array.
    """
    if max_depth < 1:
        temp = numpy.ones(mat.shape).astype(bool)
        return numpy.array(temp)

    mask = mat.copy().astype(bool)
    d = numpy.matrix(mat).astype(int)
    acc = d.copy()
    for i in range(2, max_depth + 1):
        acc *= d
        mask |= (acc == 1)
    return numpy.array(mask)


class LazyValues(object):
    """
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
    """
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


def get_coulomb_matrix(numbers, coords, alpha=1, use_decay=False):
    """
    Return the coulomb matrix for the given coords and numbers.

    C_ij = Z_i Z_j / | r_i - r_j |
    C_ii = 0.5 Z_i ** 2.4

    Parameters
    ----------
    numbers : array-like, shape=(n_atoms, )
        The atomic numbers of all the atoms

    coords : array-like, shape=(n_atoms, 3)
        The xyz coordinates of all the atoms (in angstroms)

    alpha : number, default=6
        Some value to exponentiate the distance in the coulomb matrix.

    use_decay : bool, default=False
        This setting defines an extra decay for the values as they get futher
        away from the "central atom". This is to alleviate issues the arise as
        atoms enter or leave the cutoff radius.

    Returns
    -------
    top : array, shape=(n_atoms, n_atoms)
        The coulomb matrix
    """
    top = numpy.outer(numbers, numbers).astype(numpy.float64)
    r = cdist(coords, coords)
    if use_decay:
        other = cdist([coords[0]], coords).reshape(-1)
        r += numpy.add.outer(other, other)

    r **= alpha

    with numpy.errstate(divide='ignore', invalid='ignore'):
        numpy.divide(top, r, top)
    numpy.fill_diagonal(top, 0.5 * numpy.array(numbers) ** 2.4)
    top[top == numpy.Infinity] = 0
    top[numpy.isnan(top)] = 0
    return top


def get_element_pairs(elements):
    """
    Extract all the element pairs in a molecule.

    Parameters
    ----------
    elements : list
        All the elements in the molecule

    Returns
    -------
    value : list
        All the element pairs in the molecule
    """
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
    return list(pairs.keys())


def deslugify(string):
    """
    Convert a string to a feature name and its parameters.

    Parameters
    ----------
        string : str
            The slug string to extract values from.

    Returns
    -------
        name : str
            The name of the class corresponding to the string.

        final_params : dict
            A dictonary of the feature parameters.
    """
    parts = string.split('__')
    name = parts[0]
    params = parts[1:]
    swap = {
        'None': None,
        'True': True,
        'False': False,
    }
    final_params = dict()
    for param in params:
        arg, value = param.split('=')
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        if value in swap:
            value = swap[value]
        final_params[arg] = value
    return name, final_params


def cosine_decay(R, r_cut=6.):
    r"""
    Compute all the cutoff distances.

    The cutoff is defined as

    0.5 * ( cos( \pi R_ij / R_c ) + 1, if R_ij <= R_c
    0, otherwise


    Parameters
    ----------
    R : array, shape=(N_atoms, N_atoms)
        A distance matrix for all the atoms (scipy.spatial.cdist)

    r_cut : float, default=6.
        The maximum distance allowed for atoms to be considered local to the
        "central atom".

    Returns
    -------
    values : array, shape=(N_atoms, N_atoms)
        The new distance matrix with the cutoff function applied
    """
    values = 0.5 * (numpy.cos(numpy.pi * R / r_cut) + 1)
    values[R > r_cut] = 0
    return values


def _get_form_indices(values, depth):
    """
    """
    if depth < 1:
        return [], False

    # get the first value
    for val in values:
        break
    else:
        raise ValueError("No values to use.")

    value_length = len(val)
    if depth >= value_length:
        return list(range(value_length)), False

    middle_idx = value_length // 2
    even = not (value_length % 2)
    both = even and depth % 2
    half_depth = depth // 2
    start = middle_idx - half_depth - both
    end = middle_idx + half_depth + (not even)
    res = list(range(start, end))
    if not even and not (depth % 2):
        res.remove(middle_idx)
    return res, bool(both)


def get_index_mapping(values, depth, add_unknown):
    if depth < 1:
        # Just a constant value
        return (lambda _: 0), 1, False
    extra = bool(add_unknown)
    idxs, both = _get_form_indices(values, depth)
    new_values = [tuple(x[i] for i in idxs) for x in values]
    if both:
        other_idxs = [i + 1 for i in idxs]
        temp = [tuple(x[i] for i in other_idxs) for x in values]
        new_values.extend(temp)
    new_values = set(sort_chain(x) for x in new_values)

    mapping = {key: i for i, key in enumerate(sorted(new_values))}

    def map_func(key):
        key = tuple(key[i] for i in idxs)
        if not both:
            key = sort_chain(key)
        if key not in mapping and add_unknown:
            return -1
        return mapping[key]
    return map_func, len(mapping) + extra, both


def needs_reversal(chain):
    """
    Determine if the chain needs to be reversed.

    This is to set the chains such that they are in a canonical ordering

    Parameters
    ----------
    chain : tuple
        A tuple of elements to treat as a chain

    Returns
    -------
    needs_flip : bool
        Whether or not the chain needs to be reversed
    """
    x = len(chain)
    if x == 1:
        first = 0
        second = 0
    else:
        q, r = divmod(x, 2)
        first = q - 1
        second = q + r

    while first >= 0 and second < len(chain):
        if chain[first] > chain[second]:
            # Case where order reversal is needed
            return True
        elif chain[first] == chain[second]:
            # Indeterminate case
            first -= 1
            second += 1
        else:
            # Case already in the correct order
            return False
    return False


def sort_chain(chain):
    if needs_reversal(chain):
        return chain[::-1]
    return chain


def get_angles(coords):
    r"""
    Get the angles between all triples of coords.

    The resulting values are [0, \pi] and all invalid values are nans.

    Parameters
    ----------
        coords : numpy.array, shape=(n_atoms, n_dim)
            An array of all the coordinates.

    Returns
    -------
        res : numpy.array, shape=(n_atoms, n_atoms, n_atoms)
            An array the angles of all triples.
    """
    diffs = coords - coords[:, None]
    lengths = numpy.linalg.norm(diffs, axis=2)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        unit = diffs / lengths[:, :, None]
    res = numpy.einsum('ijk,jmk->ijm', unit, unit)
    numpy.clip(res, -1., 1., res)
    numpy.arccos(res, res)
    return res
