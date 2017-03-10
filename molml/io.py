"""
A collection of functions for loading molecule data from different file types.
"""
import numpy

from .constants import ELE_TO_NUM


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
    elements : list
        All the elements in the molecule.

    numbers : list
        All the atomic numbers in the molecule.

    coords : numpy.array, shape=(n_atoms, 3)
        The atomic coordinates of the molecule.
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
    elements : list
        All the elements in the molecule.

    numbers : list
        All the atomic numbers in the molecule.

    coords : numpy.array, shape=(n_atoms, 3)
        The atomic coordinates of the molecule.
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
    elements : list
        All the elements in the molecule.

    numbers : list
        All the atomic numbers in the molecule.

    coords : numpy.array, shape=(n_atoms, 3)
        The atomic coordinates of the molecule.
    """
    elements = []
    numbers = []
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
            numbers.append(ELE_TO_NUM[ele])
            coords.append([float(x) for x in vals[2:5]])
    return elements, numbers, numpy.array(coords)
