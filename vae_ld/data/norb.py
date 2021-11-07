import gzip

import PIL
import numpy as np
import requests
from PIL import Image
from tensorflow.python.platform.gfile import GFile

from vae_ld.data import util, logger
from vae_ld.data.dataset import Data


class SmallNORB(Data):
    """SmallNORB dataset. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    The data set was first used in [2] can be downloaded from
    https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/. Images are resized to 64x64.

    The ground-truth factors of variation are:
    0 - category (5 different values)
    1 - elevation (9 different values)
    2 - azimuth (18 different values)
    3 - lighting condition (6 different values)

    The instance in each category is randomly sampled when generating the images.

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    [2] LeCun et al. (2004), Learning Methods for Generic Object Recognition with Invariance to Pose and
    Lighting. IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chunk_names = ["5x46789x9x18x6x2x96x96-training", "5x01235x9x18x6x2x96x96-testing"]
        self._file_ext = ["cat", "dat", "info"]
        self._file_template = "smallnorb-{}-{}.mat"
        self._factors_shape = (5, 10, 9, 18, 6)
        # Instances are not part of the latent space.
        self.latent_factor_indices = [0, 2, 3, 4]
        self._factors_nb = len(self._factors_shape)

        self._data, features = self.load_data()
        self.index = util.StateSpaceAtomIndex(self._factors_shape, features)
        self.state_space = util.SplitDiscreteStateSpace(self._factors_shape, self.latent_factor_indices)

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()

        list_of_images = []
        list_of_features = []
        file_path = str(self.path / self._file_template)

        for chunk_name in self._chunk_names:
            norb = _read_binary_matrix(file_path.format(chunk_name, "dat"))
            list_of_images.append(self._resize_images(norb[:, 0]))
            norb_class = _read_binary_matrix(file_path.format(chunk_name, "cat"))
            norb_info = _read_binary_matrix(file_path.format(chunk_name, "info"))
            list_of_features.append(np.column_stack((norb_class, norb_info)))

        features = np.concatenate(list_of_features, axis=0)
        features[:, 3] = features[:, 3] / 2  # azimuth values are 0, 2, 4, ..., 24
        return np.concatenate(list_of_images, axis=0), features

    def download(self):
        logger.info("Downloading smallNorb dataset. This will happen only once.")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        for chunk in self._chunk_names:
            for fe in self._file_ext:
                fname = self._file_template.format(chunk, fe)
                url = "{}/{}.gz".format(self._url, fname)
                file = str(self.path / fname)
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(file, "wb") as f:
                    f.write(gzip.decompress(response.raw.read()))

    def sample_factors(self, num, seed):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, seed)

    def sample_observations_from_factors(self, factors, seed):
        all_factors = self.state_space.sample_all_factors(factors, seed)
        indices = self.index.features_to_index(all_factors)
        return self._data[indices]

    def _resize_images(self, integer_images):
        resize_dim = self.observation_shape[:2]
        resized_images = np.zeros((integer_images.shape[0], *resize_dim))
        for i in range(integer_images.shape[0]):
            image = Image.fromarray(integer_images[i, :, :])
            image = image.resize(resize_dim,  PIL.Image.ANTIALIAS)
            resized_images[i] = image
        resized_images = np.expand_dims(resized_images.astype(np.float32), axis=3)
        return resized_images / 255.


def _read_binary_matrix(filename):
    """Reads and returns binary formatted matrix stored in filename."""
    with GFile(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])

        dtype_map = {
            507333717: "int8",
            507333716: "int32",
            507333713: "float",
            507333715: "double"
        }
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data



