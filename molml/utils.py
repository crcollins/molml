from scipy.spatial.distance import cdist


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

def get_bond_type(element1, element2, dist):
    '''
    Given a pair of elements and a distance, return the bond order between them.
    If there is no bond, return None.
    '''
    for key in TYPE_ORDER[::-1]:
        try:
            if dist < (BOND_LENGTHS[element1][key] + BOND_LENGTHS[element2][key]):
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

    connections = {}
    for i, element1 in enumerate(elements):
        for j, element2 in enumerate(elements[i+1:]):
            j += i + 1
            dist = dist_mat[i, j]
            bond_type = get_bond_type(element1, element2, dist)
            if not bond_type: continue

            # Loop over both connection directions
            # A -> B and A <- B
            for x, y in ((i, j), (j, i)):
                if x not in connections:
                    connections[x] = {}
                connections[x][y] = bond_type
    return connections