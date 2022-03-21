import math
from pathlib import Path

import numpy as np
import requests
from keras.utils.data_utils import Sequence
from vae_ld.data import logger


class Data:
    """ Generic Data class used to load and interact with a dataset based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, *, url, path, observation_shape, data_size, name, factors_shape, **kwargs):
        self._path = Path(path).expanduser()
        self._observation_shape = tuple(observation_shape)
        self._data_size = data_size
        self._factors_shape = factors_shape
        self._factors_nb = len(self._factors_shape)
        self._data = []
        self._url = url
        self.name = name

    def __getitem__(self, key):
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
        """Load data from the dataset"""
        raise NotImplementedError()

    def sample_factors(self, batch_size, seed):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(self, factors, seed):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, batch_size, seed, unique=False, flatten=False, normalised=True):
        """Sample a batch of factors Y and observations X."""
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
        """Sample a batch of observations X."""
        return self.sample(batch_size, seed)[1]

    def download(self):
        fname = self.path / self._url.split("/")[-1]
        response = requests.get(self._url, stream=True, allow_redirects=True)
        response.raise_for_status()
        with open(str(fname), 'wb') as f:
            f.write(response.raw.read())


class DataSampler(Sequence):

    def __init__(self, *, data, batch_size, validation_split=0.2, validation=False, **kwargs):
        self._data = data
        self._batch_size = batch_size
        self._validation_size = validation_split * self.data.data_size
        self._validation = validation
        self._full_len = math.ceil(self.data.data_size / self.batch_size)
        self._val_len = math.ceil(self._validation_size / self._batch_size)

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
        if self._validation_size is None:
            return self._full_len
        elif self._validation is True:
            return self._val_len
        else:
            return self._full_len - self._val_len

    def __getitem__(self, idx):
        if self._validation is True:
            idx += self._full_len - self._val_len
        logger.debug("Getting data from index {} to {}".format(idx * self.batch_size, (idx + 1) * self.batch_size))
        data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        logger.debug("Return batch of size {}".format(data.shape))
        return (data,)
