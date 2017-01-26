import numpy
from scipy.spatial.distance import cdist

from .base import BaseFeature


class AtomKernel(BaseFeature):
    '''
    Computes a kernel between molecules using atom similarity

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    gamma : float, default=1e-7
        The hyperparameter to use for the width of the gaussian kernel

    transformer : BaseFeature, default=None
        The transformer to use to convert molecules to atom-wise features. If
        this is not given, then it is assumed that the features have already
        been created and will be passed directly to fit/transform.

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
    '''
    def __init__(self, input_type='list', n_jobs=1, gamma=1e-7,
                 transformer=None, same_element=True):
        super(AtomKernel, self).__init__(input_type=input_type, n_jobs=n_jobs)
        self.gamma = gamma
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
                computational cost in half.

        Returns
        -------
            kernel : numpy.array, shape=(n_molecules_train, n_molecules_b)
                The kernel matrix between the two sets of molecules
        '''
        kernel = numpy.zeros((len(b_feats), len(self._features)))
        for i, (x, x_nums) in enumerate(zip(b_feats, b_nums)):
            zipped = zip(self._features, self._numbers)
            for j, (y, y_nums) in enumerate(zipped):
                if symmetric and j > i:
                    continue

                # Mask to make sure only elements of the same type are
                # compared
                if self.same_element:
                    mask = numpy.equal.outer(x_nums, y_nums)
                else:
                    mask = numpy.ones((len(x_nums), len(y_nums)))

                block = cdist(x, y, 'sqeuclidean')
                block *= -self.gamma
                numpy.exp(block, block)
                kernel[i, j] = (block * mask).sum()

                if symmetric:
                    kernel[j, i] = kernel[i, j]
        return kernel

    def _para_get_numbers(self, X):
        data = self.convert_input(X)
        numbers = data.numbers
        return numbers

    def fit(self, X, y=None):
        if self.transformer is None:
            self._features, self._numbers = zip(*X)
        else:
            self._features = self.transformer.fit_transform(X, y)
            self._numbers = self.map(self._para_get_numbers, X)

    def transform(self, X, y=None):
        if self.transformer is None:
            features, numbers = zip(*X)
        else:
            features = self.transformer.transform(X, y)
            numbers = self.map(self._para_get_numbers, X)
        return self.compute_kernel(features, numbers)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.compute_kernel(self._features, self._numbers,
                                   symmetric=True)
