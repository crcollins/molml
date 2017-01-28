import numpy
from scipy.spatial.distance import cdist

from .base import BaseFeature


__all__ = ["AtomKernel"]


class AtomKernel(BaseFeature):
    '''
    Computes a kernel between molecules using atom similarity

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
        The hyperparameter to use for the width of the gaussian kernel

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

    Attributes
    ----------
    self._features : numpy.array, shape=(n_mols, (n_atoms, n_features))
        A numpy array of numpy arrays (that may be different lengths) that
        stores all of the atom features for the training molecules.

    self._numbers : numpy.array, shape=(n_mols, (n_atoms))
        A numpy array of numpy arrays (that may be different lengths) that
        stores all the atomic numbers for the training atoms.

    Raises
    ------
    ValueError
        If the input_type of the transformer and the input_type keyword given
        do not match.
    '''
    def __init__(self, input_type=None, n_jobs=1, gamma=1e-7,
                 transformer=None, same_element=True):
        super(AtomKernel, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self.gamma = gamma
        if self.input_type is None:
            if transformer is not None:
                self.input_type = transformer.input_type
            else:
                # Standard default
                self.input_type = 'list'
        else:
            if transformer is not None:
                if self.input_type != transformer.input_type:
                    string = "The input_type for transformer (%r) does not "
                    string += "match the input_type of the atom kernel (%r)"
                    raise ValueError(string % (transformer.input_type,
                                               self.input_type))
            else:
                # input_type is ignored
                pass

        self.transformer = transformer
        self.same_element = same_element
        self._features = None
        self._numbers = None

    def compute_kernel(self, b_feats, b_nums, symmetric=False):
        '''
        Compute a Gaussian kernel between molecules based on atom features

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
        '''
        kernel = numpy.zeros((len(b_feats), len(self._features)))
        for i, (x, x_nums) in enumerate(zip(b_feats, b_nums)):
            zipped = zip(self._features, self._numbers)
            for j, (y, y_nums) in enumerate(zipped):
                if symmetric and j > i:
                    continue

                block = cdist(x, y, 'sqeuclidean')
                block *= -self.gamma
                numpy.exp(block, block)
                # Mask to make sure only elements of the same type are
                # compared
                if self.same_element:
                    mask = numpy.equal.outer(x_nums, y_nums)
                    block *= mask
                kernel[i, j] = block.sum()

                if symmetric:
                    kernel[j, i] = kernel[i, j]
        return kernel

    def _para_get_numbers(self, X):
        '''
        Inner parallel function to collect the atomic numbers of a molecule

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
        '''
        data = self.convert_input(X)
        numbers = data.numbers
        return numbers

    def fit(self, X, y=None):
        '''
        Fit the model

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
        '''
        if self.transformer is None:
            feats, numbers = zip(*X)
            self._features = numpy.array(feats)
            self._numbers = numpy.array(numbers)
        else:
            self._features = self.transformer.fit_transform(X, y)
            self._numbers = numpy.array(self.map(self._para_get_numbers, X))
        return self

    def transform(self, X, y=None):
        '''
        Transform features/molecules into a kernel matrix

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
        '''
        if self._features is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        if self.transformer is None:
            features, numbers = zip(*X)
        else:
            features = self.transformer.transform(X, y)
            numbers = self.map(self._para_get_numbers, X)
        return self.compute_kernel(features, numbers)

    def fit_transform(self, X, y=None):
        '''
        A slightly cheaper way of fitting and then transforming

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
        '''
        self.fit(X)
        return self.compute_kernel(self._features, self._numbers,
                                   symmetric=True)
