import re
import numpy as np


class SplitDiscreteStateSpace(object):
    """ State space with factors split between latent variable and observations.
    Based on Locatello et al. [1]
    `implementation <https://github.com/google-research/disentanglement_lib>`_

    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, factor_sizes, latent_factor_indices):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [
            i for i in range(self.num_factors)
            if i not in self.latent_factor_indices
        ]

    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)

    def sample_latent_factors(self, num, random_state):
        """ Sample a batch of latent factors Y.

        Parameters
        ----------
        num : int
            The number of examples to return
        random_state : np.random.RandomState
            The numpy random state used to generate the sample.

        Returns
        -------
        np.array
            A (n_examples, n_factors) batch of factors
        """
        factors = np.zeros(
            shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num, random_state)
        return factors

    def sample_all_factors(self, latent_factors, random_state):
        """ Sample the remaining factors based on the latent factors.

        Parameters
        ----------
        latent_factors : np.array
            A (n_examples, n_factors) batch of factors
        random_state : np.random.RandomState
            The numpy random state used to generate the sample.

        Returns
        -------
        np.array
            A (n_examples, n_factors) batch of factors
        """
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(
            shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self.latent_factor_indices] = latent_factors
        # Complete all the other factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples, random_state)
        return all_factors

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)


class StateSpaceAtomIndex(object):
    """ Index mapping from features to positions of state space atoms.
    Based on Locatello et al. [1]
    `implementation <https://github.com/google-research/disentanglement_lib>`_

    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, factor_sizes, features):
        """ Create the StateSpaceAtomIndex

        Parameters
        ----------
        factor_sizes : list
            List of integers containing the number of distinct values for each
            of the factors.
        features : np.array
            Numpy matrix where each row contains a different factor
            configuration. The matrix needs to cover the whole state space.
        """
        self.factor_sizes = factor_sizes
        num_total_atoms = np.prod(self.factor_sizes)
        self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
        feature_state_space_index = self._features_to_state_space_index(features)
        if np.unique(feature_state_space_index).size != num_total_atoms:
            raise ValueError("Features matrix does not cover the whole state space.")
        lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
        lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
        self.num_total_atoms = num_total_atoms
        self.state_space_to_save_space_index = lookup_table

    def features_to_index(self, features):
        """ Return the indices in the input space for given factor configurations.

        Parameters
        ----------
        features : np.array
            Numpy matrix where each row contains a different factor
            configuration for which the indices in the input space should be
            returned.

        Returns
        -------
        np.array
            The indices in the input space for given factor configurations.

        """
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    def index_to_features(self, indexes):
        """ Return the given factor configurations corresponding to the indices in the input space.

        Parameters
        ----------
        indexes : np.array
            The indices in the input space for given factor configurations.

        Returns
        -------
        np.array
            Numpy matrix where each row contains a different factor
            configuration corresponding to the given indices in the input space.

        """
        if np.any(indexes > self.num_total_atoms) or np.any(indexes < 0):
            raise ValueError("Indices must be within [0, {}] but {} was given".format(self.num_total_atoms, indexes))
        features = np.zeros((len(indexes), len(self.factor_bases)))
        for i, index in enumerate(indexes):
            remainder = index
            for j, fb in enumerate(self.factor_bases):
                features[i, j], remainder = np.divmod(remainder, fb)
        return features.astype(np.int64)

    def _features_to_state_space_index(self, features):
        """ Return the indices in the atom space for given factor configurations.

        Parameters
        ----------
        features : np.array
            Numpy matrix where each row contains a different factor
            configuration for which the indices in the atom space should be
            returned.

        Returns
        -------
        np.array
            The indices in the atom space for given factor configurations.

        """
        if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or
                np.any(features < 0)):
            raise ValueError("Feature indices have to be within [0, factor_size-1] but {} was given".format(features))
        return np.array(np.dot(features, self.factor_bases), dtype=np.int64)


def tryint(s):
    """ Try to convert a string to an integer. If the given string is an int, convert it to an integer, else return a string.

    Parameters
    ----------
    s : str
        The string to process

    Returns
    -------
    str or int
        The converted integer if s only contained decimal characters, otherwise the initial string

    Examples
    --------
    >>> tryint("123")
        123
    >>> tryint("123a")
        "123a"
    >>> tryint("hello")
        "hello"
    """
    try:
        return int(s)
    except ValueError:
        return s


def natural_sort(s):
    """ Natural sort implementation taken from `Ned Batchelder website <https://nedbatchelder.com/blog/200712/human_sorting.html>`_.
    Turn a string into a list of string and number chunks.

    Parameters
    ----------
    s : str
        The string to process

    Examples
    --------
    >>> natural_sort("z23a")
        ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def get_unique_samples(X):
    if isinstance(X, tuple):
        data, idx = np.unique(X[0], axis=0, return_index=True)
        labels = np.array(X[1])[idx]
        return data, labels
    return np.unique(X, axis=0)
