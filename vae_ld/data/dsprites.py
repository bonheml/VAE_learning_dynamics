import requests
import PIL.Image
from tensorflow.python.platform import gfile

from vae_ld.data import util, logger
from vae_ld.data.dataset import Data
import numpy as np


class DSprites(Data):
    """DSprites dataset. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    The data set was originally used in [2] and can be downloaded from
    https://github.com/deepmind/dsprites-dataset.

    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    [2] Higgins et al. (2017) beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
    Proceedings of ICLR 2017
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_factor_indices = list(range(6))
        self._data, self._factors_shape = self.load_data()
        self.factor_bases = np.prod(self._factors_shape) / np.cumprod(self._factors_shape)
        self.state_space = util.SplitDiscreteStateSpace(self._factors_shape, self.latent_factor_indices)

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.download()

        file_path = str(self.path / self._url.split("/")[-1])
        with gfile.Open(file_path, "rb") as data_file:
            # Data was saved originally using python2, so we need to set the encoding.
            data = np.load(data_file, encoding="latin1", allow_pickle=True)
        return np.array(data["imgs"]), np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        return self.sample_observations_from_factors_no_color(factors, random_state)

    def sample_observations_from_factors_no_color(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return np.expand_dims(self._data[indices].astype(np.float32), axis=3)

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factors_shape[i], size=num)

    def download(self):
        logger.info("Downloading Dsprites dataset. This will happen only once.")
        super().download()


class ColorDSprites(DSprites):
    """Color DSprites. Based on Locatello et al. [1] implementation and dataset
    (https://github.com/google-research/disentanglement_lib)

    This data set is the same as the original DSprites data set except that when
    sampling the observations X, the sprite is colored in a randomly sampled
    color.

    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_observations_from_factors(self, factors, random_state):
        no_color_observations = self.sample_observations_from_factors_no_color(factors, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)
        color = np.repeat(random_state.uniform(0.5, 1, [observations.shape[0], 1, 1, 3]), observations.shape[1], axis=1)
        color = np.repeat(color, observations.shape[2], axis=2)
        return observations * color


class NoisyDSprites(DSprites):
    """Noisy DSprites. Based on Locatello et al. [1] implementation and dataset
    (https://github.com/google-research/disentanglement_lib)

    This data set is the same as the original DSprites data set except that when
    sampling the observations X, the background pixels are replaced with random
    noise.

    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_observations_from_factors(self, factors, random_state):
        no_color_observations = self.sample_observations_from_factors_no_color(factors, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)
        color = random_state.uniform(0, 1, [observations.shape[0], 64, 64, 3])
        return np.minimum(observations + color, 1.)


class ScreamDSprites(DSprites):
    """Scream DSprites. Based on Locatello et al. [1] implementation and dataset
    (https://github.com/google-research/disentanglement_lib)

    This data set is the same as the original DSprites data set except that when
    sampling the observations X, a random patch of the Scream image is sampled as
    the background and the sprite is embedded into the image by inverting the
    color of the sampled patch at the pixels of the sprite.

    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scream_url = "https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg"
        self._scream = self.load_scream()

    def load_scream(self):
        file = self.path / self._scream_url.split("/")[-1]
        if not file.exists():
            self.download_scream()

        with gfile.Open(file, "rb") as f:
            scream = PIL.Image.open(f)
        scream.thumbnail((350, 274))
        return np.array(scream) * 1. / 255.

    def download_scream(self):
        logger.info("Downloading The Scream image. This will happen only once.")
        file_path = str(self.path / self._scream_url.split("/")[-1])
        response = requests.get(self._scream_url)
        response.raise_for_status()
        with open(str(file_path), 'wb') as f:
            f.write(response.content)

    def sample_observations_from_factors(self, factors, random_state):
        no_color_observations = self.sample_observations_from_factors_no_color(factors, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)

        for i in range(observations.shape[0]):
            x_crop = random_state.randint(0, self._scream.shape[0] - 64)
            y_crop = random_state.randint(0, self._scream.shape[1] - 64)
            cropped_scream = self._scream[x_crop:x_crop + 64, y_crop:y_crop + 64]
            background = (cropped_scream + random_state.uniform(0, 1, size=3)) / 2.
            mask = (observations[i] == 1)
            background[mask] = 1 - background[mask]
            observations[i] = background
        return observations
