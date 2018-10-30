import numpy as np

from molml.features import Connectivity
from molml.constants import BOND_LENGTHS

# Currently, there are two recommended ways to work with elements that are not
# included in molml/constants.py. In this example, we will look at an iron
# complex (iron is not in the constants).
# Maybe at some point, molml will include more constants, but it seems outside
# of the scope of this library.

if __name__ == '__main__':
    elements = ['Fe', 'H', 'H', 'H', 'H', 'H', 'H']
    coords = np.array([
        [0., 0., 0.],
        [1.46, 0., 0.],
        [0., 1.46, 0.],
        [0., 0., 1.46],
        [-1.46, 0., 0.],
        [0., -1.46, 0.],
        [0., 0., -1.46],
    ])
    feat = Connectivity(depth=2)
    # Notice the warning about missing elements.
    print(feat.fit_transform([(elements, coords)]))

    # 1) Modify the values in the constants module before your script.
    BOND_LENGTHS['Fe'] = {'1': 1.32}
    print(feat.fit_transform([(elements, coords)]))
    del BOND_LENGTHS['Fe']

    # 2) Include connectivity information in your data. The other instances
    # where constants are used (electronegativity, element symbols, atomic
    # numbers).
    connections = {
        0: {1: '1', 2: '1', 3: '1', 4: '1', 5: '1', 6: '1'},
        1: {0: '1'},
        2: {0: '1'},
        3: {0: '1'},
        4: {0: '1'},
        5: {0: '1'},
        6: {0: '1'},
    }
    print(feat.fit_transform([(elements, coords, connections)]))
