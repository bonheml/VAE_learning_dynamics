import glob
from zipfile import ZipFile

import PIL.Image
import numpy as np
from imageio import imread

from vae_ld.data import logger
from vae_ld.data.dataset import Data


class CelebA(Data):
    """
    CelebA dataset [1] with images rescaled to 64x64.

    References
    ----------
    .. [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. In Proceedings of
       the IEEE international conference on computer vision (pp. 3730-3738).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data = self.load_data()

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            dataset = self.download("{}/img_align_celeba.zip".format(self.path))
            dataset = self.save_images(dataset)
        else:
            self.save_images()
            dataset = self.read_files()
        return dataset

    def read_files(self):
        logger.info("Loading celeba dataset.")
        return np.load("{}/celeba.npy".format(self.path))

    def save_images(self):
        images = glob.glob("{}/img_align_celeba/*.jpg".format(self.path))
        arr = np.zeros((self.data_size, *self.observation_shape), dtype=np.float32)
        for i, img in enumerate(images):
            img = PIL.Image.open(img)
            img.thumbnail(self.observation_shape)
            arr[i] = np.array(img) / 255.
        logger.info("Saving np array of images to {}/celeba.npy".format(self.path))
        np.save("{}/celeba.npy".format(self.path), arr)
        return arr

    def download(self, fname=None):
        logger.info("Downloading celebA dataset. This will only happen once")
        super(CelebA, self).download(fname=fname)
        with ZipFile(fname) as zfile:
            zfile.extractall(path=str(self.path))
        return self.save_images()
