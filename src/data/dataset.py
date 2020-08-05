import tarfile
import shutil
from pathlib import Path
import requests
from PIL import Image
from scipy.io import loadmat
import numpy as np
from tensorflow.python.platform import gfile


class Data:
    """ Generic Data class used to load and interact with a dataset based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, *, url, path, observation_shape):
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
        factors = np.zeros(shape=(batch_size, self._factors_nb), dtype=np.int64)
        for i in range(self._factors_nb):
            factors[:, i] = np.random.RandomState(seed).randint(self.factors_shape[i], size=batch_size)
        return factors

    def sample_observations_from_factors(self, factors):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, batch_size, seed):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(batch_size, seed)
        return factors, self.sample_observations_from_factors(factors)

    def sample_observations(self, batch_size, seed):
        """Sample a batch of observations X."""
        return self.sample(batch_size, seed)[1]


class Cars3D(Data):
    """Cars3D data set. based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    The data set was first used in [2] and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.
    The ground-truth factors of variation are:
     - elevation (4 different values)
     - azimuth (24 different values)
     - object dip_type (183 different values)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    [2] Reed et al. (2015). Deep visual analogy-making. In Advances in neural information processing systems
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._factors_shape = (183, 24, 4)
        self._factors_nb = len(self._factors_shape)

    def sample_observations_from_factors(self, factors):
        """Sample a batch of observations X given a batch of factors Y."""
        reshaped_data = self.data.reshape(self.factors_shape + self.observation_shape)
        return np.array([reshaped_data[tuple(factors[i])] for i in range(len(factors))], dtype=np.float32)

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()

        all_files = [x for x in gfile.ListDirectory(str(self.path)) if ".mat" in x]
        dataset = np.zeros(self.factors_shape + self.observation_shape)
        for i, filename in enumerate(all_files):
            data_mesh = self._load_mesh(filename)
            dataset[i] = data_mesh
        dataset = dataset.reshape((-1,) + self._observation_shape)
        self._data = dataset
        return dataset

    def _load_mesh(self, filename):
        """Parses a single source file and rescales contained images."""
        file_path = str(self.path / filename)
        with gfile.Open(file_path, "rb") as f:
            mesh = np.einsum("abcde->deabc", loadmat(f)["im"])

        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((24, 4) + self.observation_shape)
        for i in range(flattened_mesh.shape[0]):
            pic = Image.fromarray(flattened_mesh[i].astype(np.uint8))
            pic.thumbnail(self.observation_shape[:2], Image.ANTIALIAS)
            rescaled_mesh[i // 4, i % 4] = np.array(pic)
        return rescaled_mesh * 1. / 255

    def download(self):
        fname = self.path / "tmp" / self._url.split("/")[-1]
        fname.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(self._url, stream=True)
        response.raise_for_status()
        with open(str(fname), 'wb') as f:
            f.write(response.raw.read())

        tfile = tarfile.open(str(fname))
        tfile.extractall(path=str(self.path / "tmp"))
        for x in gfile.ListDirectory(str(self.path / "tmp" / "data" / "cars")):
            shutil.copy(x, str(self.path.parent))

        shutil.rmtree(str(self.path / "tmp"), ignore_errors=True)
