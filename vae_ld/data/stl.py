import shutil
import tarfile

import PIL.Image
import numpy as np
import requests
from vae_ld.data import logger
from vae_ld.data.dataset import Data


class Stl(Data):
    """
    Stl10 dataset [1] with images rescaled to 64x64.

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
            dataset = self.download()
            dataset = self.save_images(dataset)
        else:
            dataset = self.read_files()
        return dataset

    def save_images(self, images):
        # Resize images before saving the numpy file
        arr = np.zeros((images.shape[0], *self.observation_shape))
        logger.debug("Resize dataset from {} to {}".format(images.shape, (images.shape[0], *self.observation_shape)))
        for i, img in enumerate(images):
            img = PIL.Image.fromarray(img)
            img.thumbnail(self.observation_shape[:2])
            arr[i] = np.array(img)
        logger.info("Saving images to {}".format(self.path))
        np.save("{}/stl.npy".format(self.path), arr)
        return arr

    def read_files(self):
        logger.info("Loading Stl dataset.")
        return np.load("{}/stl.npy".format(self.path)) / 255.

    def download(self):
        logger.info("Downloading Stl dataset. This will happen only once.")
        fname = self.path / "tmp" / self._url.split("/")[-1]
        fname.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(self._url, stream=True)
        response.raise_for_status()
        with open(str(fname), 'wb') as f:
            f.write(response.raw.read())

        logger.info("Decompressing the dataset")
        tfile = tarfile.open(str(fname))
        tfile.extractall(path=str(self.path / "tmp"))
        extraction_path = str(self.path / "tmp" / "stl10_binary" / "unlabeled_X.bin")
        logger.info("Extracting the unlabelled images.")
        with open(extraction_path, "rb") as f:
            images = np.fromfile(f, dtype=np.uint8)
        shutil.rmtree(str(self.path / "tmp"), ignore_errors=True)

        # Extract in channel first order, that is shape (3,96,96)
        images = np.reshape(images, (-1, 3, 96, 96))
        # Transpose to channel last order, that is shape (96,96,3)
        images = np.transpose(images, (0, 3, 2, 1))
        return images
