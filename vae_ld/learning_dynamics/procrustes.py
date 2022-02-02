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

    def normalise(self, X):
        # Here we use the same normalisation as in "Grounding Representation Similarity with Statistical Testing",
        # Ding et al. 2021
        x_norm = X - X.mean(axis=1, keepdims=True)
        x_norm /= np.linalg.norm(x_norm)
        return x_norm

    def procrustes(self, X, Y):
        A = self.normalise(X)
        B = self.normalise(Y)
        logger.debug("Shape of x : {}, shape of y: {}".format(A.shape, B.shape))
        A_sq_frob = np.linalg.norm(A, ord="fro") ** 2
        B_sq_frob = np.linalg.norm(B, ord="fro") ** 2
        AB_nuc = np.linalg.norm(A.T @ B, ord="nuc")
        return A_sq_frob + B_sq_frob - 2 * AB_nuc

    def __call__(self, x, y):
        return self.procrustes(x, y)
