import glob
from zipfile import ZipFile

from imageio import imread
import numpy as np
from vae_ld.data import logger
from vae_ld.data.dataset import Data
from vae_ld.data.util import CustomIndex


class SymSol(Data):
    """
    Symmetric Solids dataset [1] with duplicate images and marked shapes removed. All images are rescaled to 64x64.

    References
    ----------
    .. [1] Adam Coates, Honglak Lee, Andrew Y. Ng An Analysis of Single Layer Networks in Unsupervised Feature Learning
    AISTATS, 2011.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_names = ["tet", "cube", "icosa", "cone", "cyl"]
        self._data, self._features, self._lookup_table = self.load_data()
        self.index = CustomIndex(self._features)

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            res = self.download("symsol_reduced.zip")
        else:
            res = self.read_files()
        return res

    def save_images(self):
        images = glob.glob("{}/symsol_reduced/*.png".format(self.path))
        arr = np.zeros((self.data_size, *self.observation_shape))
        features = np.zeros((self.data_size, 1))
        for i, img in enumerate(images):
            arr[i] = np.expand_dims(imread(img), axis=-1) / 255.
            features[i] = [self.label_names.index(a) for a in self.label_names if a in img][0]
        lookup_table = [np.where(features == i)[0] for i in range(5)]
        logger.info("Saving np array of images to {}/symsol.npy".format(self.path))
        np.save("{}/symsol.npy".format(self.path), arr)
        np.save("{}/symsol_labels.npy".format(self.path), features)
        return arr, features, lookup_table

    def read_files(self):
        logger.info("Loading symsol_reduced dataset.")
        data = np.load("{}/symsol.npy".format(self.path))
        features = np.load("{}/symsol_labels.npy".format(self.path))
        lookup_table = [np.where(features == i)[0] for i in range(5)]
        return data, features, lookup_table

    def download(self, fname=None):
        logger.info("Downloading symsol_reduced dataset. This will only happen once")
        super().download()
        with ZipFile("{}/symsol_reduced.zip".format(self.path)) as zfile:
            zfile.extractall(path=str(self.path))
        return self.save_images()

    def sample_factors(self, batch_size, seed):
        return seed.choice(5, batch_size)

    def sample_observations_from_factors(self, factors, seed):
        indices = []
        for i in factors:
            indices.append(self._lookup_table[i][seed.choice(len(self._lookup_table[i]), 1)][0])
        return self._data[indices]