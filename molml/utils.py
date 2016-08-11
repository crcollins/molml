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
