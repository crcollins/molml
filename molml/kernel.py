"""
A module to compute kernel based representations.

The methods in this module are intended to be used directly as kernels for
kernel methods (e.g. SVMs or KRR). This results in features that are dependent
on the number of molecules used to fit the transformers. These should then
give single vectors that have length n_fit_molecules.
"""
from contextlib import contextmanager

import numpy
from scipy.spatial.distance import cdist

from .base import BaseFeature, InputTypeMixin


__all__ = ("AtomKernel", )

KERNELS = {
    'rbf': 'sqeuclidean',
    'laplace': 'cityblock',
}


class AtomKernel(InputTypeMixin, BaseFeature):
    """
    Computes a kernel between molecules using atom similarity.

    This kernel comes with the benefit that because it is atom-wise, it stays
    size consistent. So, for extensive properties this should properly scale
    with the size of the molecule compared to other kernel methods.

    Parameters
    ----------
    input_type : string, default=None
        Specifies the format the input values will be (must be one of 'list'
        or 'filename'). Note: This input type depends on the value from
        transformer. See Below for more details. If this value is None, then
        it will take the value from transformer, or if there is no transformer
        then it will default to 'list'. If a value is given and it does not
        match the value given for the transformer, then this will raise a
        ValueError.

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    gamma : float, default=1e-7
        The hyperparameter to use for the width of the RBF or Laplace kernels

    transformer : BaseFeature, default=None
        The transformer to use to convert molecules to atom-wise features. If
        this is not given, then it is assumed that the features have already
        been created and will be passed directly to fit/transform. Note: if no
        transformer is given, then the assumed input type is going to be a
        a list of (numbers, features) pairs. Where numbers is an iterable of
        the atomic numbers, and features is a numpy array of the features
        (shape=(n_atoms, n_features)).

    same_element : bool, default=True
        Require that the atom-atom similarity only be computed if the two
        atoms are the same element.

    kernel : string or callable, default="rbf"
        The kernel function to use when computing the atom-atom interactions.
        There possible string options are the keys of KERNELS. If a callable
        object is given, then it must take two arrays and return the pairwise
        kernel metric between them.

    Attributes
    ----------
    _features : numpy.array, shape=(n_mols, (n_atoms, n_features))
        A numpy array of numpy arrays (that may be different lengths) that
        stores all of the atom features for the training molecules.

    _numbers : numpy.array, shape=(n_mols, (n_atoms))
        A numpy array of numpy arrays (that may be different lengths) that
        stores all the atomic numbers for the training atoms.

    Raises
    ------
    ValueError
        If the input_type of the transformer and the input_type keyword given
        do not match.

    References
    ----------
    Barker, J.; Bulin, J.;  Hamaekers, J. LC-GAP: Localized Coulomb Descriptors
    for the Gaussian Approximation Potential. 2016
    """
    ATTRIBUTES = ("_features", "_numbers")
    LABELS = None

    def __init__(self, input_type=None, n_jobs=1, gamma=1e-7,
                 transformer=None, same_element=True, kernel="rbf"):
        super(AtomKernel, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self.gamma = gamma
        self.check_transformer(transformer)
        self.transformer = transformer
        self.same_element = same_element
        self.kernel = kernel
        self._features = None
        self._numbers = None

        # This makes this not thread safe for multiple calls to transform
        self._temp_other_features = None
        self._temp_other_numbers = None

    @contextmanager
    def _temp_store(self, feats, nums):
        """
        Helper method to store features/numbers on the object

        This makes the parallel kernel generation easier.

        Parameters
        ----------
        feats : numpy.array
            The array of features to use.

        nums : numpy.array
            The array of numbers to use.
        """
        self._temp_other_features = feats
        self._temp_other_numbers = nums
        yield
        self._temp_other_features = None
        self._temp_other_numbers = None

    def _para_compute_kernel(self, data):
        """
        Inner parallel function to compute molecule-molecule the kernel value.

        This is formulated in a way that it can easily be done in a map/reduce
        fashion.

        Parameters
        ----------
        X : tuple
            A tuple of ints (i, j) to index the test molecule and the train
            molecule respectively.

        Returns
        -------
        value : float
            The final resulting kernel value.

        Raises
        ------
            ValueError
                If the kernel type is not a valid input.
        """
        i, j = data
        x = self._temp_other_features[i]
        x_nums = self._temp_other_numbers[i]
        y = self._features[j]
        y_nums = self._numbers[j]

        if callable(self.kernel):
            block = self.kernel(x, y)
        elif self.kernel in KERNELS:
            string = KERNELS[self.kernel]
            block = cdist(x, y, string)
            block *= -self.gamma
            numpy.exp(block, block)
        else:
            raise ValueError("This is not a valid kernel value.")

        # Mask to make sure only elements of the same type are compared
        if self.same_element:
            mask = numpy.equal.outer(x_nums, y_nums)
            block *= mask
        return block.sum()

    def compute_kernel(self, b_feats, b_nums, symmetric=False):
        """
        Compute a kernel between molecules based on atom features.

        Parameters
        ----------
            b_feats : list of numpy.array, shape=(n_molecules_b, )
                Each array is of shape (n_atoms, n_features), where n_atoms is
                for that particular molecule.

            b_nums : list of lists, shape=(n_molecules_b, )
                Contains all the atom elements for each molecule in group b

            symmetric : bool, default=True
                Whether or not the kernel is symmetric. This is just to cut the
                computational cost in half. This is mainly an optimization
                when computing the (train, train) kernel.

        Returns
        -------
            kernel : numpy.array, shape=(n_molecules_b, n_molecules_fit)
                The kernel matrix between the two sets of molecules
        """
        kernel = numpy.zeros((len(b_feats), len(self._features)))

        with self._temp_store(b_feats, b_nums):
            if symmetric:
                idxs = numpy.tril_indices(kernel.shape[0])
            else:
                xvals = numpy.arange(len(b_feats))
                yvals = numpy.arange(len(self._features))
                vals = numpy.meshgrid(xvals, yvals)
                idxs = (vals[0].reshape(-1), vals[1].reshape(-1))

            values = self.map(self._para_compute_kernel, zip(*idxs))
            kernel[idxs] = values
            if symmetric:
                kernel[idxs[1], idxs[0]] = values

        return kernel

    def _para_get_numbers(self, X):
        """
        Inner parallel function to collect the atomic numbers of a molecule.

        This is formulated in a way that it can easily be done in a map/reduce
        fashion.

        Parameters
        ----------
        X : object
            An object to use for the fit.

        Returns
        -------
        numbers : list
            A list of the atomic numbers in the molecule.
        """
        data = self.convert_input(X)
        numbers = data.numbers
        return numbers

    def fit(self, X, y=None):
        """
        Fit the model.

        If there is no self.transformer, then this assumes that the input is
        a list of (features, numbers) pairs where features is a numpy array of
        features (shape=(n_atoms, n_features)), and numbers is a list of
        atomic numbers in the molecule.

        Otherwise, it directly passes these values to the transformer to
        compute the features, and extracts all the atomic numbers.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to fit.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.transformer is None:
            feats, numbers = zip(*X)
            self._features = numpy.array(feats)
            self._numbers = numpy.array(numbers)
        else:
            self._features = self.transformer.fit_transform(X, y)
            self._numbers = numpy.array(self.map(self._para_get_numbers, X))
        return self

    def transform(self, X, y=None):
        """
        Transform features/molecules into a kernel matrix.

        If there is no self.transformer, then this assumes that the input is
        a list of (features, numbers) pairs where features is a numpy array of
        features (shape=(n_atoms, n_features)), and numbers is a list of
        atomic numbers in the molecule.

        Otherwise, it directly passes these values to the transformer to
        compute the features, and extracts all the atomic numbers.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to transform

        Returns
        -------
        kernel : array, shape=(n_samples, n_samples_fit)
            The resulting kernel matrix

        Raises
        ------
            ValueError
                If the transformer has not been fit.
        """
        self.check_fit()
        if self.transformer is None:
            features, numbers = zip(*X)
        else:
            features = self.transformer.transform(X, y)
            numbers = self.map(self._para_get_numbers, X)
        return self.compute_kernel(features, numbers)

    def fit_transform(self, X, y=None):
        """
        A slightly cheaper way of fitting and then transforming.

        This benefit comes from the resulting kernel matrix being symmetric.
        Meaning, that only half of it has to be computed.

        Parameters
        ----------
        X : list, shape=(n_samples, )
            A list of objects to use to transform

        Returns
        -------
        kernel : array, shape=(n_samples, n_samples)
            The resulting kernel matrix
        """
        self.fit(X)
        return self.compute_kernel(self._features, self._numbers,
                                   symmetric=True)
