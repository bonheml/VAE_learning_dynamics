import math
from pathlib import Path
import numpy as np
import requests
from keras.utils.data_utils import Sequence
from sklearn.utils import extmath
import tensorflow as tf
from vae_ld.data import logger, util
from sklearn.preprocessing import OneHotEncoder


class Data:
    """ Generic Data class used to load and interact with a dataset based on Locatello et al. [1]
    `implementation <https://github.com/google-research/disentanglement_lib>`_


    Parameters
    ----------
    url : str
        The URL where the dataset can be downloaded
    path : str
        The absolute path where the dataset will be stored
    observation_shape : list
        The shape of one data example (e.g., [64,64,3])
    data_size : int
        The number of data examples in the dataset
    name : str
        The name of the dataset
    factors_shape : list
        A list containing the number of labels for each ground truth factor (e.g., [4, 24, 183] for cars3D)


    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, *, url, path, observation_shape, data_size, name, factors_shape, **kwargs):
        self._path = Path(path).expanduser()
        self._observation_shape = tuple(observation_shape)
        self._data_size = data_size
        self._factors_shape = factors_shape
        self._factors_nb = len(self._factors_shape)
        features = extmath.cartesian([np.array(list(range(i))) for i in self.factors_shape]) if self._factors_nb > 0 else None
        self.index = util.StateSpaceAtomIndex(self.factors_shape, features) if self._factors_nb > 0 else None
        self._data = []
        self._url = url
        self.name = name

    def __getitem__(self, key):
        """ Get data example at index key

        Parameters
        ----------
        key
            The index or slice of data examples to retrieve

        Returns
        -------
        np.array
            The data examples
        """
        return self._data[key]

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def data_size(self):
        return self._data_size

    @property
    def factors_shape(self):
        return self._factors_shape

    @property
    def path(self):
        return self._path

    @property
    def data(self):
        return self._data

    def load_data(self):
        """ Load data from the dataset

        Returns
        -------
        np.array
            The (data_size, observation_shape) dataset

        Raises
        -------
        NotImplementedError
        """
        raise NotImplementedError()

    def sample_factors(self, batch_size, seed):
        """ Sample a batch of factors Y.

        Parameters
        ----------
        batch_size : int
            The number of examples to return
        seed : np.random.RandomState
            The numpy random state used to generate the sample.

        Returns
        -------
        np.array
            A (n_examples, n_factors) batch of factors

        Raises
        -------
        NotImplementedError
        """
        raise NotImplementedError()

    def sample_observations_from_factors(self, factors, seed):
        """ Sample a batch of observations X given a batch of factors Y.

        Parameters
        ----------
        factors : np.array
            A (n_examples, n_factors) batch of factors

        seed : np.random.RandomState
            The numpy random state used to generate the sample.

        Returns
        -------
        np.array
            A (n_examples, observation_shape) batch of observations

        Raises
        -------
        NotImplementedError
        """
        raise NotImplementedError()

    def sample(self, batch_size, seed, unique=False, flatten=False, normalised=True):
        """ Sample a batch of factors Y and observations X.

        Parameters
        ----------
        batch_size : int
            The number of examples to return
        seed : np.random.RandomState
            The numpy random state used to generate the sample.
        unique : bool, optional
            If True remove duplicates after generation, else the sample can contain duplicate examples. Default False.
        flatten : bool, optional
            If False, any m-dimensional observation shape results in an array of m+1 dimensions.
            If True, any m-dimensional observation shape is flattened resulting in a two dimensional array. Default False.
        normalised : bool, optional
            If True, the observations are divided by 255, else, they are left as-is. Default True.

        Returns
        -------
        tuple
            A tuple containing (factors, obs)
        """
        factors = self.sample_factors(batch_size, seed)
        obs = self.sample_observations_from_factors(factors, seed)
        if unique is True:
            obs, idxs = np.unique(obs, axis=0, return_index=True)
            factors = factors[idxs]
        if flatten is True:
            obs = obs.reshape(obs.shape[0], np.prod(obs.shape[1:]))
        if normalised is False:
            obs *= 255.
        return factors, obs

    def sample_observations(self, batch_size, seed):
        """ Sample a batch of observations X.

        Parameters
        ----------
        batch_size : int
            The number of examples to return
        seed : np.random.RandomState
            The numpy random state used to generate the sample.

        Returns
        -------
        np.array
            A (batch_size, observation_shape) array of observations
        """
        return self.sample(batch_size, seed)[1]

    def download(self, fname=None):
        """ Download the dataset.

        Note
        ----
        This will only be done if the dataset does not already exists in the given path

        Returns
        -------
        None
        """
        if fname is None:
            fname = self.path / self._url.split("/")[-1]
        response = requests.get(self._url, stream=True, allow_redirects=True)
        response.raise_for_status()
        with open(str(fname), 'wb') as f:
            f.write(response.raw.read())


class DataSampler(Sequence):
    """ Sample data examples from a dataset.

    Attributes
    ----------
    data : Data
        The dataset to sample from

    batch_size : int
        The size of the batches to sample

    validation : bool or None
        Whether to generate a train or validation set.

    Parameters
    ----------
    data : Data
        The dataset to sample from
    batch_size : int
        The size of the batches to sample
    validation_split : float, optional
        The Fraction of data left for validation. Default 0.2.
    validation : bool or None, optional
        If True, a validation sets are generated according to ``validation_split``.
        If False, a train set is generated and a ratio of ``validation_split`` examples is removed from the dataset.
        If None, a train set is generated and ``validation_split`` is ignored.
        Default False.
    get_labels: int, optional
        If 0, don't return any label. The output will be (data,)
        If 1, return labels for classification task. The output will be (data, labels)
        If 2, labels are provided as input (e.g. for iVAE). The output will be ([data, labels],)
    """

    def __init__(self, data, batch_size, seed, *args, validation_split=0.2, validation=False, get_labels=0,
                 categorical=True, **kwargs):
        self._data = data
        self._batch_size = batch_size
        data_size = np.copy(self.data.data_size)
        self._validation_size = math.ceil(validation_split * data_size)
        self._random_state = np.random.default_rng(seed)
        self._validation_idxs = self._random_state.choice(data_size - 1, self._validation_size, replace=False)
        self._train_idxs = np.array(list(set(range(data_size)) - set(self._validation_idxs)))
        logger.debug("Validation size is {}".format(len(self._validation_idxs)))
        logger.debug("Train size is {}".format(len(self._train_idxs)))
        self._random_state.shuffle(self._train_idxs)
        self._validation = validation
        self._get_labels = get_labels
        self._label_idxs = None
        self._categorical = categorical
        self._categorical_encoder = None
        self._update_cat_encoder()

    def _update_cat_encoder(self):
        if self._categorical and self._label_idxs is not None:
            cats = [[i for i in range(a)] for a in np.array(self.data.factors_shape)[self._label_idxs]]
            self._categorical_encoder = OneHotEncoder(categories=cats)

    @property
    def validation_idxs(self):
        return self._validation_idxs

    @validation_idxs.setter
    def validation_idxs(self, idxs):
        self._validation_idxs = idxs

    @property
    def train_idxs(self):
        return self._train_idxs

    @train_idxs.setter
    def train_idxs(self, idxs):
        self._train_idxs = idxs

    @property
    def labels_idxs(self):
        return self._label_idxs

    @labels_idxs.setter
    def labels_idxs(self, idxs):
        self._label_idxs = idxs
        self._update_cat_encoder()

    @property
    def y_true(self):
        idxs = self.validation_idxs if self._validation else self._train_idxs
        labels = self.data.index.index_to_features(idxs)
        if self._label_idxs is not None:
            labels = labels[:, self._label_idxs]
        return self._categorical_encoder.fit_transform(labels).toarray() if self._categorical else labels

    @property
    def labels(self):
        return self._get_labels

    @labels.setter
    def labels(self, n):
        self._get_labels = n

    @property
    def categorical(self):
        return self.categorical

    @categorical.setter
    def categorical(self, is_cat):
        self._categorical = is_cat

    @property
    def data(self):
        return self._data

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def validation(self):
        return self._validation

    @validation.setter
    def validation(self, validation):
        self._validation = validation

    def __len__(self):
        if self._validation is True:
            return len(self._validation_idxs) // self.batch_size
        return len(self._train_idxs) // self.batch_size

    def __getitem__(self, idx):
        logger.debug("Validation is {}, current index is {}".format(self.validation, idx))
        idxs_map = self._validation_idxs if self.validation else self._train_idxs
        start_idx = idx * self.batch_size
        stop_idx = (idx + 1) * self.batch_size
        if stop_idx >= len(idxs_map):
            stop_idx = len(idxs_map) - 1

        logger.debug("Retrieving indexes in range ({},{})".format(start_idx, stop_idx))
        idxs = idxs_map[start_idx:stop_idx]
        logger.debug("Getting data from indexes {}".format(idxs))
        data = self.data[idxs]
        if isinstance(data[0], tuple):
            data = [tf.convert_to_tensor(d, dtype=tf.float32) for d in zip(*data)]
            data_shape = data[0].shape
            data = list(zip(*data))
        else:
            data = tf.convert_to_tensor(self.data[idxs], dtype=tf.float32)
            data_shape = data.shape
        if self._get_labels < 2:
            logger.debug("Return batch of size {}".format(data_shape))
        if self._get_labels > 0:
            labels = self.data.index.index_to_features(idxs)
            if self._label_idxs is not None:
                labels = labels[:, self._label_idxs]
            if self._categorical is True:
                labels = self._categorical_encoder.fit_transform(labels).toarray()
            logger.debug("Factors for indexes {}: {}".format(idxs, labels))
            if self._get_labels == 2:
                # When the labels are used as input, we normalise their values between 0 and 1 using min-max
                # feature scaling. This way, they are in the same range as input images.
                logger.debug("Return batch of size [{}, {}]".format(data_shape, labels.shape))
                labels = tf.convert_to_tensor(labels, dtype=tf.float32)
                lmin, lmax = tf.reduce_min(labels, axis=0, keepdims=True), tf.reduce_max(labels, axis=0, keepdims=True)
                labels = (labels - lmin) / (lmax - lmin)
                return (data, labels),
            return data, labels
        return data,
