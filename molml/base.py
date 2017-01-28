import inspect
import multiprocessing
from functools import reduce

import numpy
from pathos.multiprocessing import ProcessingPool as Pool

from .utils import LazyValues, read_file_data


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
    input_type : string or list, default='list'
        Specifies the format the input values will be (must be one of 'list',
        'filename', or a list of strings). If it is a list of strings, the
        strings tell the order of (and if they are included) the different
        molecule attributes (coords, elements, numbers, connections).

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.
    '''
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
        '''
        Set the feature parameter values

        Parameters
        ----------
            kwargs : kwargs
                Key value pairs to set for the feature parameters. Keys that
                are not valid parameters will be ignored.
        '''
        for key, value in kwargs.items():
            try:
                getattr(self, key)
                setattr(self, key, value)
            except AttributeError:
                continue

    def get_params(self):
        '''
        Get a dictonary of all the feature parameters

        Returns
        -------
            params : dict
                A dictonary of all the feature parameters.
        '''
        argspec = inspect.getargspec(type(self).__init__)
        # Delete the only non-keyword argument
        args = [x for x in argspec.args if x != "self"]
        values = [getattr(self, x) for x in args]
        return {key: value for key, value in zip(args, values)}

    def slugify(self):
        '''
        Converts an instance to a simple string.

        Returns
        -------
        string : str
            The slug string

        '''
        name = type(self).__name__
        # Skip the first two parameters
        params = self._get_param_strings()[2:]
        string = '__'.join([name] + params).replace("'", '')
        return string

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
            zn)]. This allows allows for connections to be incldued. This is a
            dictonary where the keys are the indices of the atoms and the
            values are dictonaries with the key being another index and the
            value is the bond order (one of '1', 'Ar', '2', or '3').
            Example for methane
            {
                0: {1: "1", 2: "1", 3: "1", 4: "1"},
                1: {0: "1"},
                2: {0: "1"},
                3: {0: "1"},
                4: {0: "1"},
            }

            If input_type is 'filename', then it must be an iterable of
            paths/filenames for each molecule. The files must then be of the
            form
            ele1 x1 y1 z1
            ele2 x2 y2 z2
            ...
            elen xn yn zn

            If input_type is a list, then they will be treated as labels to
            each of the arguments passed in via a tuple. For example,
            input_type="list" can be reproduced with ["elements", "coords"]
            or ["elements", "coords", "connections"].

        Returns
        -------
        values : Object
            An object that allows the lazy evaluation of different properties

        Raises
        ------
        ValueError
            If the input_type given is not allowed.
        '''
        connections = None
        if self.input_type == "list":
            try:
                elements, coords = X
            except ValueError:
                elements, coords, connections = X
            values = LazyValues(elements=elements, coords=coords,
                                connections=connections)
        elif self.input_type == "filename":
            elements, numbers, coords = read_file_data(X)
            values = LazyValues(elements=elements, numbers=numbers,
                                coords=coords)
        elif type(self.input_type) in (list, tuple):
            d = {x: y for x, y in zip(self.input_type, X)}
            values = LazyValues(**d)
        else:
            raise ValueError("The input_type '%s' is not allowed." %
                             self.input_type)
        return values

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
            return list(map(f, seq))
        else:
            n_jobs = self.n_jobs

        pool = Pool(n_jobs)
        results = list(pool.map(f, seq))
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
            pairs = [(f, x, y) for x, y in zip(seq[::2], seq[1::2])]
            temp_seq = self.map(_func_star, pairs)
            # If the sequence length is odd add the last element on
            # This is because it will not get included with the zip
            if len(seq) % 2:
                temp_seq.append(seq[-1])
            seq = temp_seq
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


class MultiFeature(BaseFeature):
    '''
    A helper class to make merging different features together easier.

    Parameters
    ----------

    features : list
        This is a list of initialized feature objects to use. All of these
        features should have the standard fit/transform methods implemented.
    '''
    def __init__(self, features=None):
        self.features = features

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
        for feat in self.features:
            feat.fit(X, y)
        return self

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
        results = [feat.transform(X) for feat in self.features]
        return numpy.hstack(results)
