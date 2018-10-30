"""
A collection of functions for loading molecule data from different file types.

Note: Functions in this file should be agnostic to the elements/numbers. This
should be deferred to the LazyValues object.
"""
from .utils import LazyValues


def read_file_data(path):
    """
    Determine the file type and call the correct parser.

    The accepted file types are .out and .xyz files.

    Parameters
    ----------
    path : str
        A path to a file to read

    Returns
    -------
    elements : list
        All the elements in the molecule.

    numbers : list
        All the atomic numbers in the molecule.

    coords : numpy.array, shape=(n_atoms, 3)
        The atomic coordinates of the molecule.
    """
    end = path.split('.')[-1]
    mapping = {
        'out': read_out_data,
        'xyz': read_xyz_data,
        'mol2': read_mol2_data,
        'cry': read_cry_data,
    }
    if end in mapping:
        return mapping[end](path)
    else:
        raise ValueError("Unknown file type")


def read_out_data(path):
    """
    Read an out and extract the molecule's geometry.

    The file should be in the format::

        ele0 x0 y0 z0
        ele1 x1 y1 z1
        ...

    Parameters
    ----------
    path : str
        A path to a file to read

    Returns
    -------
    val : LazyValues
        An object storing all the data
    """
    elements = []
    coords = []
    with open(path, 'r') as f:
        for line in f:
            ele, x, y, z = line.strip().split()
            point = (float(x), float(y), float(z))
            elements.append(ele)
            coords.append(point)
    return LazyValues(elements=elements, coords=coords)


def read_xyz_data(path):
    """
    Read an xyz file and extract the molecule's geometry.

    The file should be in the format::

        num_atoms
        comment
        ele0 x0 y0 z0
        ele1 x1 y1 z1
        ...

    Parameters
    ----------
    path : str
        A path to a file to read

    Returns
    -------
    val : LazyValues
        An object storing all the data
    """
    elements = []
    coords = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            ele, x, y, z = line.strip().split()
            point = (float(x), float(y), float(z))
            elements.append(ele)
            coords.append(point)
    return LazyValues(elements=elements, coords=coords)


def read_mol2_data(path):
    """
    Read a mol2 file and extract the molecule's geometry.

    Roughly, the file format is something like::

        @<TRIPOS>MOLECULE
        ...
        @<TRIPOS>ATOM
         1 ele0id x0 y0 z0 ele0.type 1 MOL charge0
         2 ele1id x1 y1 z1 ele1.type 1 MOL charge1
        ...
        @<TRIPOS>BOND
        ...

    Parameters
    ----------
    path : str
        A path to a file to read

    Returns
    -------
    val : LazyValues
        An object storing all the data
    """
    elements = []
    coords = []
    with open(path, 'r') as f:
        start = False
        for line in f:
            if "@<TRIPOS>ATOM" in line:
                start = True
                continue
            if "@<TRIPOS>BOND" in line:
                break  # can't use connection info yet
            if not start:
                continue
            vals = line.split()
            ele = vals[5].split('.')[0]
            elements.append(ele)
            coords.append([float(x) for x in vals[2:5]])
    return LazyValues(elements=elements, coords=coords)


def read_cry_data(path):
    """
    Read a cry file and extract the molecule's geometry.

    The format should be as follows::

        U_xx U_xy U_xz
        U_yx U_yy U_yz
        U_zx U_zy U_zz
        energy (or comment, this is ignored for now)
        ele0 x0 y0 z0
        ele1 x1 y1 z1
        ...
        elen xn yn zn

    Where the U matrix is made of the unit cell basis vectors as column
    vectors.

    Parameters
    ----------
    path : str
        A path to a file to read

    Returns
    -------
    val : LazyValues
        An object storing all the data
    """
    unit = []
    coords = []
    elements = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                unit.append([float(x) for x in parts])
            if len(parts) == 4:
                elements.append(parts[0])
                coords.append([float(x) for x in parts[1:]])
    return LazyValues(elements=elements, coords=coords, unit_cell=unit)
