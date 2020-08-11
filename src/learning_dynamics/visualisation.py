import logging

import imageio
import pandas as pd
from src.learning_dynamics.svcca import compute_pwcca
from src.learning_dynamics.utils import load_layers_from_file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class AbstractVisualiser:
    def __init__(self, *, files, layer_names=None):
        self._files = list(files)
        self._layer_names = list(layer_names) if layer_names is not None else "all"

    @property
    def files(self):
        return self._files

    @property
    def layer_names(self):
        return self._layer_names

    def _compute_model_pwcca(self, activations_1, activations_2):
        """ Compute pwcca between two models activations.
        The arguments can be activations from the same model at different time points or from different models at the same
        time point.

        :param activations_1: the activation values of the first model in a np.array of shape (nb_datapoint, nb_neurons)
        :type activations_1: numpy.array
        :param activations_2: the activation values of the second model in a np.array of shape (nb_datapoint, nb_neurons)
        :type activations_2: numpy.array
        :return: the pwcca of each layer in a np.array of shape (nb_layers,)
        :rtype: numpy.array
        """
        res = pd.DataFrame(index=self._layer_names, columns=self._layer_names)
        for i, a1 in enumerate(activations_1):
            a1 = np.mean(np.moveaxis(a1, -1, 0), axis=(1, 2)) if len(a1.shape) == 4 else np.moveaxis(a1, -1, 0)
            a1_idx = self._layer_names[i]
            for j, a2 in enumerate(activations_2):
                a2_idx = self._layer_names[j]
                a2 = np.mean(np.moveaxis(a2, -1, 0), axis=(1, 2)) if len(a2.shape) == 4 else np.moveaxis(a2, -1, 0)
                if a1.shape[0] > a2.shape[0]:
                    res[a1_idx][a2_idx] = compute_pwcca(a2, a1, epsilon=1e-10)[0]
                else:
                    res[a1_idx][a2_idx] = compute_pwcca(a1, a2, epsilon=1e-10)[0]
        return res


class SingleModelVisualiser(AbstractVisualiser):
    def __init__(self, *args, **kwargs):
        super(SingleModelVisualiser, self).__init__(*args, **kwargs)
        self._trained = load_layers_from_file(self.files[-1], layers_name=self.layer_names).values()

    def _compute_model_pwcca(self, activation_file_1, **kwargs):
        """ Compute pwcca between model activations at time t and at the end of training.

        :param **kwargs:
        :param activation_file_1: the hdf5 file containing the activation values of the model at time t
        :return: the pwcca of each layer
        """
        acts_1 = load_layers_from_file(activation_file_1, layers_name=self.layer_names).values()
        return super(SingleModelVisualiser, self)._compute_model_pwcca(acts_1, self._trained)

    @staticmethod
    def compute_heatmap(pwcca_data, png_file):
        fig, ax = plt.subplots()
        sns.heatmap(pwcca_data, ax=ax, vmin=0, vmax=1)
        plt.savefig(png_file, bbox_inches='tight')
        return imageio.imread(png_file)

    def run(self, gif_file):
        images = []
        nb_files = len(self.files)
        base_png_file = gif_file.replace(".gif", "")
        for i, file in enumerate(self.files[:-1]):
            logger.info("Computing pwcca for {} ({}/{})...".format(file, i+1, nb_files))
            res = self._compute_model_pwcca(file).astype("float64")
            logger.info("Generating graph...")
            png_file = "{}_frame_{:02d}.png".format(base_png_file, i + 1)
            images.append(self.compute_heatmap(res, png_file))
        imageio.mimsave(gif_file, images)

