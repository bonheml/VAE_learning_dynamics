import glob
from zipfile import ZipFile
import PIL.Image
import numpy as np
import pandas as pd
from vae_ld.data import logger
from vae_ld.data.dataset import Data
from vae_ld.data.util import CustomIndex


class CelebA(Data):
    """ CelebA dataset [1] with images rescaled to 64x64.

    References
    ----------
    .. [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. In Proceedings of
       the IEEE international conference on computer vision (pp. 3730-3738).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data, self._features = self.load_data()
        self.index = CustomIndex(self._features)
        self._factors_shape = [2 for _ in range(40)]

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            dataset = self.download("{}/img_align_celeba.zip".format(self.path))
        else:
            dataset = self.save_images()
            # dataset = self.read_files()
        return dataset

    def read_files(self):
        logger.info("Loading celeba dataset.")
        return np.load("{}/celeba.npy".format(self.path)), np.load("{}/celeba_labels.npy".format(self.path))

    def save_images(self):
        images = glob.glob("{}/img_align_celeba/*.jpg".format(self.path))
        df = pd.read_csv("{}/list_attr_celeba.txt".format(self.path), delim_whitespace=True).replace(-1, 0)
        df = df.rename_axis("img").reset_index()
        arr = np.zeros((self.data_size, *self.observation_shape), dtype=np.float32)
        y = np.zeros((self.data_size, 40), dtype=np.float32)
        for i, img_name in enumerate(images):
            img = PIL.Image.open(img_name)
            img.thumbnail(self.observation_shape[:2])
            img = PIL.ImageOps.pad(img, self.observation_shape[:2])
            arr[i] = np.array(img) / 255.
            idx = df.index[df["img"] == img_name.split("/")[-1]].to_list()[0]
            y[i] = df.iloc[idx].drop("img").values.astype(int)
        logger.info("Saving np array of images to {}/celeba.npy".format(self.path))
        np.save("{}/celeba.npy".format(self.path), arr)
        np.save("{}/celeba_labels.npy".format(self.path), y)
        return arr, y

    def download(self, fname=None):
        logger.info("Downloading celeba dataset. This will only happen once")
        super(CelebA, self).download(fname=fname)
        attr_url = "https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing&resourcekey=0-YW2qIuRcWHy_1C2VaRGL3Q"
        super().download(fname=attr_url)
        with ZipFile(fname) as zfile:
            zfile.extractall(path=str(self.path))
        return self.save_images()
