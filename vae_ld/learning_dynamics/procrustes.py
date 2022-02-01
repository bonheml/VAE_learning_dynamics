import numpy as np

from vae_ld.learning_dynamics import logger


class Procrustes:
    """ Computes Procrustes distance between representations x and y
    Taken from https://github.com/js-d/sim_metric/blob/main/dists/scoring.py
    Implementation of Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
    """

    def __init__(self, name="procrustes"):
        self._name = name

    @property
    def name(self):
        return self._name

    def normalise(self, x):
        # Here we use the same normalisation as in "Grounding Representation Similarity with Statistical Testing",
        # Ding et al. 2021
        x_norm = x - x.mean(axis=1, keepdims=True)
        x_norm /= np.linalg.norm(x_norm)
        return x_norm

    def procrustes(self, x, y):
        a = self.normalise(x)
        b = self.normalise(y)
        logger.debug("Shape of x : {}, shape of y: {}".format(a.shape, b.shape))
        a_sq_frob = np.sum(a ** 2)
        b_sq_frob = np.sum(b ** 2)
        nuc = np.linalg.norm(a @ b.T, ord="nuc")
        return a_sq_frob + b_sq_frob - 2 * nuc

    def __call__(self, x, y):
        return self.procrustes(x, y)
