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

    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, files):
        self._files = files

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

    def compute_and_plot_cka(self, f1, act1, f2, act2):
        res = self._compute_model_cka(act1, act2).astype("float64")
        res.to_csv("{}_{}.tsv".format(f1, f2), sep="\t")
        logger.info("Generating graph...")
        png_file = "{}_{}.png".format(f1, f2)
        self.compute_heatmap(f1, f2, res, png_file)

    def visualise(self):
        fp1 = sorted(self.files)[0]
        f1 = fp1.split("/")[-1].replace(".hdf5", "")
        logger.info("Loading activations from {}".format(f1))
        act1 = load_layers_from_file(fp1)
        logger.info("Computing self CKA of {}".format(f1))
        self.compute_and_plot_cka(f1, act1, f1, act1)
        num_files = len(self._files)
        for i, fp2 in enumerate(self._files):
            if fp2 == fp1:
                continue
            f2 = fp2.split("/")[-1].replace(".hdf5", "")
            logger.info("Loading activations from {}".format(f2))
            act2 = load_layers_from_file(fp2)
            logger.info("({}/{}) Computing CKAs between {} and {}".format(i + 1, num_files, f1, f2))
            self.compute_and_plot_cka(f1, act1, f2, act2)
            logger.info("Computing self CKA of {}".format(f2))
            self.compute_and_plot_cka(f2, act2, f2, act2)
