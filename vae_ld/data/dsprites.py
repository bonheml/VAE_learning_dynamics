import requests
import PIL
from tensorflow.python.platform import gfile
from vae_ld.data import util, logger
from vae_ld.data.dataset import Data
import numpy as np

from vae_ld.data.util import CustomIndex


class DSprites(Data):
    """ DSprites dataset. Based on Locatello et al. [1] `implementation <https://github.com/google-research/disentanglement_lib>`_.
    The `dataset <https://github.com/deepmind/dsprites-dataset>`_  was originally used in [2].

    The ground-truth factors of variation are (in the default setting):
        * 0 - shape (3 different values)
        * 1 - scale (6 different values)
        * 2 - orientation (40 different values)
        * 3 - position x (32 different values)
        * 4 - position y (32 different values)

    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    .. [2] Higgins et al. (2017) beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
           Proceedings of ICLR 2017
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_factor_indices = list(range(self._factors_nb))
        self._data = self.load_data()
        self.state_space = util.SplitDiscreteStateSpace(self.factors_shape, self.latent_factor_indices)

    def __getitem__(self, key):
        data = super().__getitem__(key)
        return self._postprocess(data)

    def _postprocess(self, data, random_state=None):
        imgs = np.array(data).astype(np.float32)
        imgs = np.expand_dims(imgs, axis=3)
        if imgs[0].shape[:2] != self.observation_shape[:2]:
            logger.info("resizing images")
            imgs = self._resize_images(imgs)
        return imgs

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()

        file_path = str(self.path / self._url.split("/")[-1])
        with gfile.Open(file_path, "rb") as data_file:
            # Data was saved originally using python2, so we need to set the encoding.
            data = np.load(data_file, encoding="latin1", allow_pickle=True)
        return data["imgs"]

    def _resize_images(self, integer_images):
        resize_dim = self.observation_shape[:2]
        integer_images = integer_images.astype(np.uint8) * 255
        resized_images = np.zeros((integer_images.shape[0], *resize_dim))
        for i in range(integer_images.shape[0]):
            image = PIL.Image.fromarray(integer_images[i, :, :])
            image.thumbnail(resize_dim)
            resized_images[i, :, :] = np.asarray(image)
        resized_images = resized_images.astype(np.float32) / 255.
        # Prevent any non-zero value to be lower than 1 after rescaling the image
        resized_images[resized_images > 0] = 1.
        return resized_images

    def sample_factors(self, num, random_state):
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = self.index.features_to_index(all_factors)
        return self._postprocess(self._data[indices], random_state)

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factors_shape[i], size=num)

    def download(self, fname=None):
        logger.info("Downloading Dsprites dataset. This will happen only once.")
        super().download()


class ColorDSprites(DSprites):
    """ Color DSprites. Based on Locatello et al. [1] `implementation <https://github.com/google-research/disentanglement_lib>`_.

    This data set is the same as the original DSprites data set except that when
    sampling the observations X, the sprite is colored in a randomly sampled
    color.

    The ground-truth factors of variation are (in the default setting):
        * 0 - shape (3 different values)
        * 1 - scale (6 different values)
        * 2 - orientation (40 different values)
        * 3 - position x (32 different values)
        * 4 - position y (32 different values)

    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._random_state = np.random.RandomState(0)

    def _postprocess(self, data, random_state=None):
        imgs = super()._postprocess(data, random_state)
        if random_state is None:
            random_state = self._random_state
        observations = np.repeat(imgs, 3, axis=3)
        color = np.repeat(random_state.uniform(0.5, 1, [observations.shape[0], 1, 1, 3]), observations.shape[1], axis=1)
        color = np.repeat(color, observations.shape[2], axis=2)
        return observations * color


class NoisyDSprites(DSprites):
    """Noisy DSprites. Based on Locatello et al. [1] `implementation and dataset <https://github.com/google-research/disentanglement_lib>`_.

    This data set is the same as the original DSprites data set except that when
    sampling the observations X, the background pixels are replaced with random
    noise.

    The ground-truth factors of variation are (in the default setting):
        * 0 - shape (3 different values)
        * 1 - scale (6 different values)
        * 2 - orientation (40 different values)
        * 3 - position x (32 different values)
        * 4 - position y (32 different values)

    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._random_state = np.random.RandomState(0)

    def _postprocess(self, data, random_state=None):
        imgs = super()._postprocess(data, random_state)
        if random_state is None:
            random_state = self._random_state
        observations = np.repeat(imgs, 3, axis=3)
        color = random_state.uniform(0, 1, [observations.shape[0], *self.observation_shape])
        return np.minimum(observations + color, 1.)


class GreyDSprites(DSprites):
    """ Grey DSprites.

    This data set is the same as the original DSprites data set except that when
    sampling the observations X, the background pixels are turned to a configurable uniform shade of grey.

    The ground-truth factors of variation are (in the default setting):
        * 0 - shape (3 different values)
        * 1 - scale (6 different values)
        * 2 - orientation (40 different values)
        * 3 - position x (32 different values)
        * 4 - position y (32 different values)
    """

    def __init__(self, grey_shade=0.8, **kwargs):
        super().__init__(**kwargs)
        self._grey_shade = grey_shade

    def _postprocess(self, data, random_state=None):
        imgs = super()._postprocess(data, random_state)
        imgs[imgs == 0.] = self._grey_shade
        return imgs


class ScreamDSprites(DSprites):
    """ Scream DSprites. Based on Locatello et al. [1] `implementation and dataset <https://github.com/google-research/disentanglement_lib>`_.

    This data set is the same as the original DSprites data set except that when
    sampling the observations X, a random patch of the Scream image is sampled as
    the background and the sprite is embedded into the image by inverting the
    color of the sampled patch at the pixels of the sprite.

    The ground-truth factors of variation are (in the default setting):
        * 0 - shape (3 different values)
        * 1 - scale (6 different values)
        * 2 - orientation (40 different values)
        * 3 - position x (32 different values)
        * 4 - position y (32 different values)

    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scream_url = "https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg"
        self._scream = None
        self._random_state = np.random.RandomState(0)

    def _postprocess(self, data, random_state=None):
        imgs = super()._postprocess(data, random_state)
        if random_state is None:
            random_state = self._random_state
        if self._scream is None:
            self._scream = self.load_scream()

        observations = np.repeat(imgs, 3, axis=3)

        for i in range(observations.shape[0]):
            crop_shape_x, crop_shape_y = self.observation_shape[0], self.observation_shape[1]
            x_crop = random_state.randint(0, self._scream.shape[0] - crop_shape_x)
            y_crop = random_state.randint(0, self._scream.shape[1] - crop_shape_y)
            cropped_scream = self._scream[x_crop:x_crop + crop_shape_x, y_crop:y_crop + crop_shape_y]
            background = (cropped_scream + random_state.uniform(0, 1, size=3)) / 2.
            mask = (observations[i] == 1)
            background[mask] = 1 - background[mask]
            observations[i] = background

        return observations

    def load_scream(self):
        """ Load `The scream` image.

        Returns
        -------
        np.array
            An array of size (350,274,3) containing a normalised version of `The scream`
        """
        file = self.path / self._scream_url.split("/")[-1]
        if not file.exists():
            self.download_scream()

        with gfile.Open(file, "rb") as f:
            scream = PIL.Image.open(f)
        scream.thumbnail((350, 274))
        return np.asarray(scream) * 1. / 255.

    def download_scream(self):
        """ Download `The scream` image.

        Note
        ----
        This will only be done if the image does not already exists in the given path.

        Returns
        -------
        None
        """
        logger.info("Downloading The Scream image. This will happen only once.")
        file_path = str(self.path / self._scream_url.split("/")[-1])
        response = requests.get(self._scream_url)
        response.raise_for_status()
        with open(str(file_path), 'wb') as f:
            f.write(response.content)


class CorrelatedBlueDSprites(Data):
    """ DSprites dataset from [1] with squares both in white and blue.
    Other shapes are only in white.

        The ground-truth factors of variation are (in the default setting):
            * 0 - colour (2 different values, 0:white, 2:blue)
            * 1 - shape (3 different values)
            * 2 - scale (6 different values)
            * 3 - orientation (40 different values)
            * 4 - position x (32 different values)
            * 5 - position y (32 different values)


        References
        ----------
        .. [1] Higgins et al. (2017) beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
               Proceedings of ICLR 2017
        """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data, self._features = self.load_data()
        self.index = CustomIndex(self._features)

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()
        elif not (self.path / "{}.npz".format(self.name)).exists():
            self.save_images()
        return self.read_files()

    def read_files(self):
        logger.info("Loading mixed dSprites dataset.")
        data = np.load("{}/{}.npz".format(self.path, self.name))
        return data["imgs"], data["latents_classes"]

    def save_images(self):
        file_path = str(self.path / self._url.split("/")[-1])
        with gfile.Open(file_path, "rb") as data_file:
            # Data was saved originally using python2, so we need to set the encoding.
            data = np.load(data_file, encoding="latin1", allow_pickle=True)
        imgs, labels = self._preprocess(data["imgs"], data["latents_classes"])
        logger.info("Saving results to {}/{}.npz".format(self.path, self.name))
        np.savez_compressed("{}/{}".format(self.path, self.name), imgs=imgs, latents_classes=labels)

    def _load_and_resize(self, imgs):
        imgs = np.array(imgs).astype(np.float32)
        imgs = np.repeat(np.expand_dims(imgs, axis=3), 3, axis=3)
        resize_dim = self.observation_shape[:2]
        resize = imgs[0].shape[:2] != resize_dim

        if resize:
            imgs = imgs.astype(np.uint8) * 255
            resized_images = np.zeros((imgs.shape[0], *resize_dim), dtype=np.float32)
            for i in range(imgs.shape[0]):
                resized_images[i] = self._resize(imgs[i], resize_dim)
        else:
            resized_images = imgs

        return resized_images

    def _preprocess(self, imgs, labels):
        imgs = self._load_and_resize(imgs)

        max_i = (labels[labels[:, 1] == 0]).shape[0]
        blue_labels = np.copy(labels[0:max_i])
        blue_labels[:, 0] = 1
        blue_squares = np.copy(imgs[0:max_i])
        blue_squares[:, :, :, :-1] = 0

        labels = np.vstack((blue_labels, labels))
        imgs = np.vstack((imgs, blue_squares))

        return imgs, labels

    def _resize(self, img, resize_dim):
        image = PIL.Image.fromarray(img)
        image.thumbnail(resize_dim)
        return np.asarray(image, dtype=np.float32) / 255.

    def download(self, fname=None):
        logger.info("Downloading dSprites dataset. This will happen only once.")
        super().download()
        return self.save_images()


class BlueDSprites(CorrelatedBlueDSprites):
    """ DSprites dataset from [1] with a subset of random shapes both in white and blue.

            The ground-truth factors of variation are (in the default setting):
                * 0 - colour (2 different values, 0:white, 2:blue)
                * 1 - shape (3 different values)
                * 2 - scale (6 different values)
                * 3 - orientation (40 different values)
                * 4 - position x (32 different values)
                * 5 - position y (32 different values)


            References
            ----------
            .. [1] Higgins et al. (2017) beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
            Framework Proceedings of ICLR 2017
    """
    def _preprocess(self, imgs, labels, n=3):
        imgs = self._load_and_resize(imgs)

        idxs = np.random.choice(labels.shape[0], labels.shape[0] // n, replace=False)
        blue_labels = np.copy(labels[idxs])
        blue_labels[:, 0] = 1
        blue_shapes = np.copy(imgs[idxs])
        blue_shapes[:, :, :, :-1] = 0

        labels = np.vstack((blue_labels, labels))
        imgs = np.vstack((imgs, blue_shapes))

        return imgs, labels


class BlueDSpritesSmall(BlueDSprites):
    """ DSprites dataset from [1] with a small subset of random shapes both in white and blue.

            The ground-truth factors of variation are (in the default setting):
                * 0 - colour (2 different values, 0:white, 2:blue)
                * 1 - shape (3 different values)
                * 2 - scale (6 different values)
                * 3 - orientation (40 different values)
                * 4 - position x (32 different values)
                * 5 - position y (32 different values)


            References
            ----------
            .. [1] Higgins et al. (2017) beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
            Framework Proceedings of ICLR 2017
    """
    def _preprocess(self, imgs, labels, n=10):
        return super(BlueDSpritesSmall, self)._preprocess(imgs, labels, n)


class BlueDSpritesFull(BlueDSprites):
    def _preprocess(self, imgs, labels, n=1):
        return super(BlueDSpritesFull, self)._preprocess(imgs, labels, n)

