"""
A collection of all the base transformer constructions.

This module is a collection of all the base classes and mixins for use with
the other transformers.
"""
import inspect
import multiprocessing
from functools import reduce
import json

import numpy
from pathos.multiprocessing import ProcessingPool as Pool

from .utils import LazyValues
from .io import read_file_data


def _func_star(args):
    """
    A function and argument expanding helper function.

    The first item in args is callable, and the remainder of the items are
    used as expanded arguments. This is to make the function header for reduce
    the same for the normal and parallel versions. Otherwise, the functions
    would have to do their own unpacking of arguments.
    """
    f = args[0]
    args = args[1:]
    return f(*args)


class BaseFeature(object):
    """
    A base class for all the features.

    Parameters
    ----------
    input_type : str, list of str, or callable, default='list'
        Specifies the format the input values will be (must be one of 'list',
        'filename', a list of strings, or a callable). If it is a list of
        strings, the strings tell the order of (and if they are included) the
        different molecule attributes (coords, elements, numbers,
        connections). If a callable is given, then it is assumed to return a
        LazyValues object.

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.
    """
    def __init__(self, input_type='list', n_jobs=1):
        self.input_type = input_type
        self.n_jobs = n_jobs

    def _get_param_strings(self):
        argspec = inspect.getargspec(type(self).__init__)
        # Delete the only non-keyword argument
        args = [x for x in argspec.args if x != "self"]
        values = [getattr(self, x) for x in args]
        return ["%s=%r" % (x, y) for x, y in zip(args, values)]

    def __repr__(self):
        name = type(self).__name__
        params = self._get_param_strings()
        return "%s(%s)" % (name, ', '.join(params))

    def set_params(self, **kwargs):
        """
        Set the feature parameter values.

        Parameters
        ----------
            kwargs : kwargs
                Key value pairs to set for the feature parameters. Keys that
                are not valid parameters will be ignored.
        """
        for key, value in kwargs.items():
            try:
                getattr(self, key)
                setattr(self, key, value)
            except AttributeError:
                continue

    def get_params(self):
        """
        Get a dictonary of all the feature parameters.

        Returns
        -------
            params : dict
                A dictonary of all the feature parameters.
        """
        argspec = inspect.getargspec(type(self).__init__)
        # Delete the only non-keyword argument
        args = [x for x in argspec.args if x != "self"]
        values = [getattr(self, x) for x in args]
        return {key: value for key, value in zip(args, values)}

    def slugify(self):
        """
        Convert an instance to a simple string.

        Returns
        -------
        string : str
            The slug string

        """
        name = type(self).__name__
        # Skip the first two parameters
        params = self._get_param_strings()[2:]
        string = '__'.join([name] + params).replace("'", '')
        return string

    def convert_input(self, X):
        """
        Convert the input (as specified in self.input_type) to a usable form.

        Parameters
        ----------
        X : list or string (depends on the instance value of input_type)
            An object that stores the data for a single molecule. See the
            Notes for more details.

        Returns
        -------
        values : Object
            An object that allows the lazy evaluation of different properties

        Raises
        ------
        ValueError
            If the input_type given is not allowed.

        Notes
        -----
        If input_type is 'list', then it must be an iterable of (elements,
        coodinates pairs) for each molecule. Where the elements are an
        iterable of the form (ele1, ele2, ..., elen) and coordinates are an
        iterable of the form [(x1, y1, z1), (x2, y2, z2), ..., (xn, yn,
        zn)]. This allows allows for connections to be included. This is a
        dictionary where the keys are the indices of the atoms and the
        values are dictonaries with the key being another index and the
        value is the bond order (one of '1', 'Ar', '2', or '3').
        Example for methane::

            {
                0: {1: "1", 2: "1", 3: "1", 4: "1"},
                1: {0: "1"},
                2: {0: "1"},
                3: {0: "1"},
                4: {0: "1"},
            }

        If input_type is 'filename', then it must be an iterable of
        paths/filenames for each molecule. Currently, the supported formats
        are: xyz, mol2, and a simple xyz format (.out).

        If input_type is a list, then they will be treated as labels to
        each of the arguments passed in via a tuple. For example,
        input_type="list" can be reproduced with ["elements", "coords"]
        or ["elements", "coords", "connections"].

        If input_type is a callable, then it is assumed that the callable
        returns a LazyValues object.
        """
        connections = None
        if self.input_type == "list":
            try:
                elements, coords = X
            except ValueError:
                elements, coords, connections = X
            values = LazyValues(elements=elements, coords=coords,
                                connections=connections)
        elif self.input_type == "filename":
            values = read_file_data(X)
        elif type(self.input_type) in (list, tuple):
            d = {x: y for x, y in zip(self.input_type, X)}
            values = LazyValues(**d)
        elif callable(self.input_type):
            values = self.input_type(X)
        else:
            raise ValueError("The input_type '%s' is not allowed." %
                             self.input_type)
        return values

    def get_labels(self):
        """
        Get the labels for the features in the transformer

        Returns
        -------
        values : tuple
            All of the labels of the resulting features.
            Note: These may not be a one-to-one mapping, but rather the order
            in which they occur.
        """
        if self.LABELS is None:
            return tuple()

        values = [tuple(sorted(getattr(self, x))) for x in self.LABELS]
        return sum(values, tuple())

    def check_fit(self):
        """
        Check if the transformer has been fit

        Raises
        ------
        ValueError
            The transformer has not been fit.
        """
        if self.ATTRIBUTES is None:
            return

        msg = "This %s instance is not fitted yet. Call 'fit' first."
        for key in self.ATTRIBUTES:
            if getattr(self, key) is None:
                raise ValueError(msg % type(self).__name__)

    @classmethod
    def get_citation(self):
        try:
            docs = self.__doc__
            idx = docs.index("References")
            values = [x.strip() for x in docs[idx:].split('\n')[2:]]
            new_values = [[]]
            for value in values:
                if "----" in value:
                    new_values.pop()
                    break
                elif not value:
                    new_values.append([])
                else:
                    new_values[-1].append(value)
            strings = [' '.join(x) for x in new_values if x]
            return '\n'.join(strings)
        except ValueError:
            return "MolML https://github.com/crcollins/molml"

    def map(self, f, seq):
        """
        Parallel implementation of map.

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
        """
        if self.n_jobs < 1:
            n_jobs = multiprocessing.cpu_count()
        elif self.n_jobs == 1:
            return list(map(f, seq))
        else:
            n_jobs = self.n_jobs

        pool = Pool(n_jobs)
        results = list(pool.map(f, seq))
        # Closing/joining is not really allowed because pathos sees pools as
        # lasting for the duration of the program.
        return results

    def reduce(self, f, seq):
        """
        Parallel implementation of reduce.

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
        """
        if self.n_jobs == 1:
            return reduce(f, seq)

        while len(seq) > 1:
            pairs = [(f, x, y) for x, y in zip(seq[::2], seq[1::2])]
            temp_seq = self.map(_func_star, pairs)
            # If the sequence length is odd add the last element on
            # This is because it will not get included with the zip
            if len(seq) % 2:
                temp_seq.append(seq[-1])
            seq = temp_seq
        return seq[0]

    def fit(self, X, y=None):
        """
        Fit the model.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        raise NotImplementedError

    def _para_transform(self, X):
        """
        A single instance of the transform procedure.

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
        """
        raise NotImplementedError

    def transform(self, X, y=None):
        """
        Framework for a potentially parallel transform.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to transform

        Returns
        -------
        array : array, shape=(n_samples, n_features)
            The transformed features
        """
        results = self.map(self._para_transform, X)
        return numpy.array(results)

    def fit_transform(self, X, y=None):
        """
        A naive default implementation of fitting and transforming.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit and then transform

        Returns
        -------
        array : array, shape=(n_samples, n_features)
            The transformed features
        """
        self.fit(X, y)
        return self.transform(X, y)

    def to_json(self):
        """
        Return model data as a json compatible dict

        This will recursively convert other transformer objects as well.

        Returns
        -------
        data : dict
            The json data
        """
        attributes = {}
        if self.ATTRIBUTES is not None:
            attributes = {key: getattr(self, key) for key in self.ATTRIBUTES}

        full_name = self.__module__ + '.' + self.__class__.__name__
        params = {}
        for key, value in self.get_params().items():
            try:
                params[key] = value.to_json()
            except AttributeError:
                params[key] = value

        data = {
                "transformer": full_name,
                "parameters": params,
                "attributes": attributes,
        }
        return data

    def save_json(self, f):
        """
        Save the model data in a json file

        Parameters
        ----------
        f : str or file descriptor
            The path to save the data or a file descriptor to save it to.
        """
        data = self.to_json()
        try:
            json.dump(data, f)
        except AttributeError:
            with open(f, 'w') as out_file:
                json.dump(data, out_file)


class SetMergeMixin(object):
    """
    A simple mixin that will merge sets.

    This mixin replaces all the duplicate code that just merges sets when
    doing the parallel fits. For this to work, it requires that the subclasses
    define `ATTRIBUTES`.
    """
    def fit(self, X, y=None):
        """
        Fit the model.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        res = self.map(self._para_fit, X)
        if len(self.ATTRIBUTES) > 1:
            temp = self.reduce(lambda x, y: tuple(set(xx) | set(yy)
                                                  for xx, yy in zip(x, y)),
                               res)
            for attr, vals in zip(self.ATTRIBUTES, temp):
                setattr(self, attr, set(vals))
        else:
            vals = self.reduce(lambda x, y: set(x) | set(y), res)
            setattr(self, self.ATTRIBUTES[0], set(vals))
        return self


class InputTypeMixin(object):
    """
    A simple mixin to to check input_types if there are multiples.

    This mixin adds a method to check if a transformer parameter does not have
    the same input_type as the parent object.
    """
    def check_transformer(self, transformer):
        """
        Check a transformer.

        Parameters
        ----------
        transformer : BaseFeature
            A transformer object.

        Raises
        ------
        ValueError
            If the input_type pairing given is not allowed.
        """
        if self.input_type is None:
            if transformer is not None:
                self.input_type = transformer.input_type
            else:
                # Standard default
                self.input_type = 'list'
        elif transformer is not None:
            if self.input_type != transformer.input_type:
                string = "The input_type for transformer (%r) does not "
                string += "match the input_type of this %s (%r)"
                raise ValueError(string % (transformer.input_type,
                                           self.__class__.__name__,
                                           self.input_type))
