import glob
from pathlib import Path

import imageio
import pandas as pd

from vae_ld.learning_dynamics.cka import CKA
from vae_ld.learning_dynamics.utils import load_layers_from_file
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import seaborn as sns

from vae_ld.learning_dynamics import logger


class Visualiser:
    def __init__(self, input_dir):
        self._files = glob.glob(str(Path(input_dir) / "*.hdf5"))
        self.cka = CKA()
        self.activations = {}

    @property
    def files(self):
        return self._files

    def _compute_model_cka(self, model_1, model_2):
        """ Compute cka between two models activations.
        The arguments can be activations from the same model at different time points or from different models at the same
        time point.

        :param model_1: the layers and activations of the first model
        :type model_1: dict
        :param model_2: the layers and activations of the second model
        :type model_2: dict
        :return: the CKA of each layer in a np.array of shape (nb_layers,)
        :rtype: numpy.array
        """
        res = {}
        for m1_layer, m1_activations in model_1.items():
            res[m1_layer] = {}
            for m2_layer, m2_activations in model_2.items():
                logger.info("Computing CKA({}, {})".format(m1_layer, m2_layer))
                res[m1_layer][m2_layer] = self.cka(m1_activations, m2_activations)
        return pd.DataFrame(res)

    @staticmethod
    def compute_heatmap(f1, f2, cka_data, png_file):
        fig, ax = plt.subplots()
        sns.heatmap(cka_data, ax=ax, vmin=0, vmax=1)
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        plt.savefig(png_file, bbox_inches='tight')
        return imageio.imread(png_file)

    def visualise(self):
        files = list(combinations_with_replacement(self._files, 2))
        num_files = len(files)
        for i, (fp1, fp2) in enumerate(files):
            f1 = fp1.split("/")[-1].replace(".hdf5", "")
            f2 = fp2.split("/")[-1].replace(".hdf5", "")
            logger.info("Loading activations from files")
            if f1 not in self.activations.keys():
                self.activations[f1] = load_layers_from_file(fp1)
            if f2 not in self.activations.keys():
                self.activations[f2] = load_layers_from_file(fp2)
            logger.info("({}/{}) Computing CKAs between {} and {}".format(i + 1, num_files, f1, f2))
            res = self._compute_model_cka(self.activations[f1], self.activations[f2]).astype("float64")
            res.to_csv("{}_{}.tsv".format(f1, f2, i + 1), sep="\t")
            logger.info("Generating graph...")
            png_file = "{}_{}.png".format(f1, f2, i + 1)
            self.compute_heatmap(f1, f2, res, png_file)
