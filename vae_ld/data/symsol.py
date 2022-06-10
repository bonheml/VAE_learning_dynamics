import glob
from zipfile import ZipFile

from imageio import imread
import numpy as np
from vae_ld.data import logger
from vae_ld.data.dataset import Data


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
        self._data = self.load_data()

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            dataset = self.download("symsol_reduced.zip")
        else:
            dataset = self.read_files()
        return dataset

    def save_images(self):
        images = glob.glob("{}/symsol_reduced/*.png".format(self.path))
        arr = np.zeros((self.data_size, *self.observation_shape))
        for i, img in enumerate(images):
            arr[i] = np.expand_dims(imread(img), axis=-1) / 255.
        logger.info("Saving np array of images to {}/symsol.npy".format(self.path))
        np.save("{}/symsol.npy".format(self.path), arr)
        return arr

    def read_files(self):
        logger.info("Loading symsol_reduced dataset.")
        return np.load("{}/symsol.npy".format(self.path))

    def download(self, fname=None):
        logger.info("Downloading symsol_reduced dataset. This will only happen once")
        super().download()
        with ZipFile("{}/symsol_reduced.zip".format(self.path)) as zfile:
            zfile.extractall(path=str(self.path))
        return self.save_images()
