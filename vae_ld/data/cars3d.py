from sklearn.utils import extmath
import tarfile
import shutil
import requests
from PIL import Image
from scipy.io import loadmat
import numpy as np
from tensorflow.python.platform import gfile

from vae_ld.data import util, logger
from vae_ld.data.dataset import Data


class Cars3D(Data):
    """Cars3D data set. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    The data set was first used in [2] and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.
    The ground-truth factors of variation are:
     - elevation (4 different values)
     - azimuth (24 different values)
     - object type (183 different values)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    [2] Reed et al. (2015). Deep visual analogy-making. In Advances in neural information processing systems
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_factor_indices = [0, 1, 2]
        features = extmath.cartesian([np.array(list(range(i))) for i in self.factors_shape])
        self.index = util.StateSpaceAtomIndex(self.factors_shape, features)
        self.state_space = util.SplitDiscreteStateSpace(self._factors_shape,
                                                        self.latent_factor_indices)
        self._data = self.load_data()

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, seed):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, seed)
        indices = self.index.features_to_index(all_factors)
        return self.data[indices].astype(np.float32)

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()

        all_files = [x for x in gfile.ListDirectory(str(self.path)) if ".mat" in x]
        if len(all_files) == 0:
            raise FileNotFoundError("The given data directory is empty. Remove it to download the dataset properly "
                                    "during next call.")
        dataset = np.zeros((24 * 4 * 183, *self.observation_shape))
        for i, filename in enumerate(all_files):
            data_mesh = self._load_mesh(filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i, len(factor1) * len(factor2))
            ])
            indexes = self.index.features_to_index(all_factors)
            dataset[indexes] = data_mesh
        dataset = dataset.reshape((-1,) + self._observation_shape)
        return dataset

    def _load_mesh(self, filename):
        """Parses a single source file and rescales contained images."""
        file_path = str(self.path / filename)
        with gfile.Open(file_path, "rb") as f:
            logger.debug("loading {}".format(file_path))
            mesh = np.einsum("abcde->deabc", loadmat(f)["im"])

        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((flattened_mesh.shape[0],) + self.observation_shape)
        for i in range(flattened_mesh.shape[0]):
            pic = Image.fromarray(flattened_mesh[i, :, :, :].astype(np.uint8))
            pic.thumbnail(self.observation_shape[:2], Image.ANTIALIAS)
            rescaled_mesh[i, :, :, :] = np.array(pic)
        return rescaled_mesh * 1. / 255

    def download(self):
        logger.info("Downloading Cars3D dataset. This will happen only once.")
        fname = self.path / "tmp" / self._url.split("/")[-1]
        fname.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(self._url, stream=True)
        response.raise_for_status()
        with open(str(fname), 'wb') as f:
            f.write(response.raw.read())

        tfile = tarfile.open(str(fname))
        tfile.extractall(path=str(self.path / "tmp"))
        extraction_path = str(self.path / "tmp" / "data" / "cars")
        for x in gfile.ListDirectory(extraction_path):
            shutil.copy("{}/{}".format(extraction_path, x), str(self.path))

        shutil.rmtree(str(self.path / "tmp"), ignore_errors=True)
