"""
A collection of asssorted utility functions.
"""
from builtins import range
import importlib
import json
import warnings
from itertools import product

import numpy
from scipy.spatial.distance import cdist
from scipy.special import expit
import scipy.stats

from .constants import ELE_TO_NUM, NUM_TO_ELE, TYPE_ORDER, BOND_LENGTHS


def lerp(a, b, x):
    return a * (1 - x) + b * x


def lerp_smooth(x):
    span = x[1] - x[0]
    temp = numpy.zeros(x.shape)
    mask = numpy.logical_and(0 <= x, x <= span)
    idx = numpy.where(mask)[0]
    val = x[mask] / span
    temp[idx] = val
    try:
        temp[idx + 1] = 1 - val
    except IndexError:
        pass
    return temp


SMOOTHING_FUNCTIONS = {
    "norm_cdf": scipy.stats.norm.cdf,
    "zero_one": lambda x: (x > 0.).astype(float),
    "expit": expit,
    "tanh": lambda x: (numpy.tanh(x)+1) / 2,
    "norm": scipy.stats.norm.pdf,
    "expit_pdf": scipy.stats.logistic.pdf,
    "spike": lambda x: (numpy.abs(x) < 1.).astype(float),
    "lerp": lerp_smooth,
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

    Parameters
    ----------
    element1 : str
        The element of the first atom
    element2 : str
        The element of the second atom
    dist : float
        The distance between the two atoms
    Returns
    -------
    key : str
        The type of the bond
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
    Return a dictionary edge list.

    Each value is is a tuple of the index of
    the connecting atom and the bond order as a string. Where the bond order
    is one of ['1', 'Ar', '2', '3'].

    Example HCN::

        {
            0: {1: '1', 2: '3'},
            1: {0: '1'},
            2: {0: '3'},
        }


    Parameters
    ----------
    elements : list
        All the elements in the molecule.

    coords : array, shape=(n_atoms, 3)
        The coordinates of the atoms in the molecule.

    Returns
    -------
    connections : dict, int->dict
        Contains all atoms that are connected to each atom and bond type.
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


def get_depth_threshold_mask_connections(connections, max_depth=1):
    """
    Get the depth threshold mask from connections.

    Parameters
    ----------
    connections : dict, index->list of indices
        A dictionary that contains lists of all connected atoms.

    max_depth : int, default=1
        The maximum depth to allow in the masking

    Returns
    -------
    mask : numpy.array, shape=(len(connections), len(connections))
        A mask of all the atoms that are less than or equal to `max_depth`
        away.
    """
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

    Parameters
    ----------
    mat : numpy.array, shape=(n, n)
        A connectivity matrix where element (i,j) is true if (i,j) has a bond.
    max_depth : int, default=1
        The maximum number of steps to look for connectivity

    Returns
    -------
    mask : numpy.array, shape=(n, n)
        A boolean mask indicating if from point i you can get to point j.
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
        A dictionary edge table with all the bidirectional connections.

    numbers : array-like, shape=(n_atoms, ), default=None
        The atomic numbers of all the atoms.

    coords : array-like, shape=(n_atoms, 3), default=None
        The xyz coordinates of all the atoms (in angstroms).

    elements : array-like, shape=(n_atoms, ), default=None
        The element symbols of all the atoms.

    unit_cell : array-like, shape=(3, 3), default=None
        An array of unit cell basis vectors, where the vectors are columns.


    Attributes
    ----------
    connections : dict, key->list of keys
        A dictionary edge table with all the bidirectional connections. If the
        initialized value for this was None, then this will be computed from
        the coords and numbers/elements.

    numbers : array, shape=(n_atoms, )
        The atomic numbers of all the atoms. If the initialized value for this
        was None, then this will be computed from the elements.

    coords : array, shape=(n_atoms, 3)
        The xyz coordinates of all the atoms (in angstroms).

    elements : array, shape=(n_atoms, )
        The element symbols of all the atoms. If the initialized value for this
        was None, then this will be computed from the numbers.

    unit_cell : array, shape=(3, 3)
        An array of unit cell basis vectors, where the vectors are columns.
    """
    def __init__(self, connections=None, coords=None, numbers=None,
                 elements=None, unit_cell=None):
        self._connections = connections
        self._coords = self._none_check(coords)
        self._numbers = self._none_check(numbers)
        self._elements = self._none_check(elements)
        self._unit_cell = self._none_check(unit_cell)
        self.__crystal_size = None

    def _none_check(self, x):
        return numpy.array(x) if x is not None else x

    def fill_in_crystal(self, radius=None, units=None):
        coords = numpy.array(self.coords)
        offsets = list(_radial_iterator(self.unit_cell, radius))
        self.__crystal_size = len(offsets)

        new_coords = []
        for offset in offsets:
            new_coords.append(coords + offset)
        self._coords = numpy.concatenate(new_coords)

        if self._numbers is not None:
            self._numbers = numpy.tile(self._numbers, self.__crystal_size)

        if self._elements is not None:
            self._elements = numpy.tile(self._elements, self.__crystal_size)

        if self._connections is not None:
            new_conn = {}
            n = len(self._connections)
            for key, items in self._connections.items():
                for i in range(self.__crystal_size):
                    off = n * i
                    values = {inner_key + off: value for
                              inner_key, value in items.items()}
                    new_conn[key + off] = values
            self._connections = new_conn
            # TODO: Add in geoms

    @property
    def connections(self):
        if self._connections is None:
            self._connections = get_connections(self.elements, self.coords)
        return self._connections

    @property
    def unit_cell(self):
        if self._unit_cell is None:
            raise ValueError("No unit cell exists.")
        return self._unit_cell

    @property
    def coords(self):
        if self._coords is None:
            raise ValueError("No coordinates exist.")
        return self._coords

    @property
    def numbers(self):
        if self._numbers is None:
            if self._elements is not None:
                temp = [ELE_TO_NUM[x] for x in self._elements]
                self._numbers = numpy.array(temp)
            else:
                raise ValueError("No elements to convert to numbers.")
        return self._numbers

    @property
    def elements(self):
        if self._elements is None:
            if self._numbers is not None:
                temp = [NUM_TO_ELE[x] for x in self._numbers]
                self._elements = numpy.array(temp)
            else:
                raise ValueError("No numbers to convert to elements.")
        return self._elements


def get_coulomb_matrix(numbers, coords, alpha=1, use_decay=False):
    r"""
    Return the coulomb matrix for the given coords and numbers.

    .. math::

        C_{ij} = \begin{cases}
            \frac{Z_i Z_j}{\| r_i - r_j \|^\alpha} & i \neq j\\
            \frac{1}{2} Z_i^{2.4} & i = j
        \end{cases}

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
            A dictionary of the feature parameters.
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

    .. math::

        f_{R_{c}}(R_{ij}) = \begin{cases}
            0.5 ( \cos( \frac{\pi R_{ij}}{R_c} ) + 1 ), & R_{ij} \le R_c \\
            0,  & otherwise
        \end{cases}


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
    """
    Determine the ordering and mapping of feature groups.

    Parameters
    ----------
    values : list
        A list of possible values.

    depth : int
        The number of elements to use from each values value.

    add_unknown : bool
        Whether or not to include an extra collector for unknown values.

    Returns
    -------
    map_func : function(key)->int
        A function that gives the mapping index for a given key.

    length : int
        The length of the mapping values.

    both : bool
        Indicates whether both values are needed in a loop (A, B) vs (B, A).
    """
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
    """
    Sort a chain from the inside out.

    Parameters
    ----------
    chain : tuple
        A tuple of elements to treat as a chain

    Returns
    -------
    chain : tuple
        The sorted chain
    """
    if needs_reversal(chain):
        return chain[::-1]
    return chain


def get_angles(coords):
    r"""
    Get the angles between all triples of coords.

    The resulting values are :math:`[0, \pi]` and all invalid values are NaNs.

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


def load_json(f):
    """
    Load the model data from a json file

    Parameters
    ----------
    f : str or file descriptor
        The path to save the data or a file descriptor to save it to.

    Returns
    -------
    obj : Transformer
        The transformer object.
    """
    try:
        data = json.load(f)
    except AttributeError:
        with open(f, 'r') as in_file:
            data = json.load(in_file)
    module, klass = data["transformer"].rsplit('.', 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Throw away the warnings in Python 3.x because of Dill...
        # https://github.com/uqfoundation/dill/issues/210
        m = importlib.import_module(module)

    cls = getattr(m, klass)
    obj = cls(**data["parameters"])
    for key, value in data["attributes"].items():
        setattr(obj, key, value)
    return obj


def _radial_iterator(X, r_max):
    X = numpy.array(X)
    norm = numpy.linalg.norm
    lengths = norm(X, axis=0)
    # Compute the upper bounds for each axis
    steps = numpy.ceil(r_max / lengths).astype(int)
    ranges = [range(-x, x + 1) for x in steps]

    for group in product(*ranges):
        group = numpy.array(group)
        temp_z = numpy.dot(X, group)
        if norm(temp_z) > r_max:
            continue
        yield temp_z
