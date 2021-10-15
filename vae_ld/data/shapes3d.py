import h5py
import numpy as np

from vae_ld.data import util, logger
from vae_ld.data.dataset import Data


class Shapes3D(Data):
    """Shapes3D dataset. Based on Locatello et al. [1] implementation
      (https://github.com/google-research/disentanglement_lib)
  
    The data set was first used in [2] and can be downloaded from
    https://storage.cloud.google.com/3d-shapes/3dshapes.h5
  
    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)
    
    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    [2] Kim and Mnih (2018). Disentangling by Factorising.
    In Proceedings of the 35th International Conference on Machine Learning (ICML)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._h5py_dataset, self._data = self.load_data()
        self._factors_shape = [10, 10, 10, 8, 4, 15]
        self.latent_factor_indices = list(range(6))
        self._factors_nb = len(self._factors_shape)
        self.state_space = util.SplitDiscreteStateSpace(self._factors_shape, self.latent_factor_indices)
        self.factor_bases = np.prod(self._factors_shape) / np.cumprod(self._factors_shape)

    def __getitem__(self, key):
        # Beware, because of the limitations of h5py, this will only work for indexes and contiguous slicing.
        return self._data[key].astype(np.float32) / 255.

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()

        fname = str(self.path / self._url.split("/")[-1])
        data = h5py.File(fname, 'r')
        imgs = data["images"]
        return data, imgs

    def download(self):
        logger.info("Downloading Shapes3D dataset. This will happen only once.")
        super().download()

    def sample_factors(self, num, seed):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, seed)

    def sample_observations_from_factors(self, factors, seed):
        data = []
        all_factors = self.state_space.sample_all_factors(factors, seed)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        for i in indices:
            data.append(self._data[i].astype(np.float32) / 255.)
        return np.array(data)

    def __del__(self):
        self._h5py_dataset.close()
