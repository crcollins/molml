"""
A module to compute fragment based representations.

This module contains a variety of methods to extract features from molecules
based on defined fragments in the molecule. This means that every molecule will
result in an array of values (n_fragments, n_features). Note: If atom-wise
features are used, then this would extend to be (n_fragments, n_atoms,
n_features).
"""
import os
import glob
from functools import partial
from builtins import range

import numpy
import six

from .base import BaseFeature


__all__ = ("FragmentMap", )


def _glob_search(label, search_dirs):
    for d in search_dirs:
        string = os.path.join(d, label + '.*')
        found = sorted(glob.glob(string))
        if found:
            return found[0]
    else:
        raise ValueError('Label ("%s") not found in search dirs.' % label)


class FragmentMap(BaseFeature):
    """
    Extract information based on features from fragments.

    This is like if there were `n` features that were extracted from the
    molecule of interest, and each of these `n` features corresponded to their
    own feature vectors. These fragments are then used together as a single
    representation. The output of these fragment vectors is in the same order
    that they are given.

    For example,

        FragmentMap().fit_transform([['A', 'B'], ['C, 'A'], ['B', 'C'])

    would produce arrays like

        [[f_A, f_B], [f_C, f_A], [f_B, f_C]]

    for a final shape of (3, 2, n_features).

    Parameters
    ----------
    input_type : str, default='filename'
        Specifies the format the input values will be (must be one of 'label'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    transformer : BaseFeature, default=None
        Some feature extractor that takes inputs and converts them to a numpy
        array of data. This should convert the fragment fragments into some
        vector representation to use. Because the information given to this
        class is at the label/filename level, the transformer must be able to
        work with the filenames directly. Either using the standard 'filename'
        `input_type`, or using a user-defined function.

    filename_to_label : callable or str, default='basename'
        The function to use to convert labels into filenames. The function
        should take a single str argument and return a label to use for that
        filename. The conversion between labels and filenames is not really
        required, but may allow for simpler bookkeeping outside this class.
        There are some predefined functions available in cls.LABEL_FUNCTIONS
        ('identity', 'basename') as recommendations for what to use.

    label_to_filename : callable or list of str, default=('.', )
        A function to convert labels into filenames to pass to the transformer.
        The function should take a single str argument and return a valid path.
        If a valid path does not exist, this should raise a ValueError.
        If this is a list, then it will be interpreted as a list of paths to
        search for files. Specifically, these are used in globs of the form
        os.path.join(dir_name, label + '.*'). Note: This will only use the
        first file that is found matching that label. The directories will be
        searched in the order given.

    Attributes
    ----------
    _x_fragments : dict, str->numpy.array
        Dictionary mapping label strings to their corresponding feature
        vectors.
    """
    ATTRIBUTES = ('_x_fragments', )
    LABELS = (('get_mapping_labels', None), )
    LABEL_FUNCTIONS = {
        'identity': lambda x: x,
        'basename': lambda x: os.path.splitext(os.path.basename(x))[0],
    }

    def __init__(self, input_type='filename', n_jobs=1, transformer=None,
                 filename_to_label='basename', label_to_filename=('.', )):
        super(FragmentMap, self).__init__(input_type=input_type, n_jobs=n_jobs)
        if transformer is None:
            raise ValueError('transformer can not be None.')
        self.transformer = transformer
        self.filename_to_label = filename_to_label
        self.label_to_filename = label_to_filename
        self._x_fragments = None

    def fit(self, X, y=None):
        unique_values = set(numpy.reshape(X, -1))
        filenames, labels = self.convert_input(unique_values)
        x_ligands = self.transformer.fit_transform(filenames)
        self._x_fragments = {x: y for x, y in zip(labels, x_ligands)}
        return self

    def _get_filename_to_label(self):
        if callable(self.filename_to_label):
            return self.filename_to_label
        else:
            return self.LABEL_FUNCTIONS[self.filename_to_label]

    def _get_label_to_filename(self):
        if callable(self.label_to_filename):
            return self.label_to_filename
        else:
            return partial(_glob_search, search_dirs=self.label_to_filename)

    def convert_input(self, X):
        if self.input_type == 'filename':
            func = self._get_filename_to_label()
            labels = [func(x) for x in X]
            filenames = X
        elif self.input_type == 'label':
            filenames = []
            func = self._get_label_to_filename()
            for x in X:
                try:
                    filenames.append(func(x))
                except ValueError:
                    pass
            labels = X
        else:
            m = 'This only accepts "filename" or "label" for input_type.'
            raise ValueError(m)
        return tuple(filenames), tuple(labels)

    def _lookup(self, fragment):
        return self._x_fragments[fragment]

    def _para_transform(self, X):
        self.check_fit()
        _, labels = self.convert_input(X)
        return numpy.array([self._lookup(x) for x in labels])

    def get_mapping_labels(self):
        try:
            labels = self.transformer.get_labels()
        except AttributeError:
            # Hack to get length of features
            length = len(six.next(six.itervalues(self._x_fragments)))
            labels = [str(x) for x in range(length)]
        return labels
