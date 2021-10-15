from pathlib import Path
import requests


class Data:
    """ Generic Data class used to load and interact with a dataset based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, *, url, path, observation_shape, **kwargs):
        self._path = Path(path).expanduser()
        self._observation_shape = tuple(observation_shape)
        self._factors_shape = None
        self._factors_nb = None
        self._data = None
        self._url = url

    @property
    def observation_shape(self):
        return self._observation_shape

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

    def sample(self, batch_size, seed):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(batch_size, seed)
        return factors, self.sample_observations_from_factors(factors, seed)

    def sample_observations(self, batch_size, seed):
        """Sample a batch of observations X."""
        return self.sample(batch_size, seed)[1]

    def download(self):
        fname = self.path / self._url.split("/")[-1]
        response = requests.get(self._url, stream=True, allow_redirect=True)
        response.raise_for_status()
        with open(str(fname), 'wb') as f:
            f.write(response.raw.read())





