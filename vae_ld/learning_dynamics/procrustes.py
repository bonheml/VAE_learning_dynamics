import numpy as np
import tensorflow as tf
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
        X_norm = X - X.mean(axis=1, keepdims=True)
        X_norm /= np.linalg.norm(X_norm)
        return X_norm

    def procrustes(self, X, Y):
        A = self.normalise(X)
        B = self.normalise(Y)
        logger.debug("Shape of x : {}, shape of y: {}".format(A.shape, B.shape))
        A_sq_frob = np.linalg.norm(A, ord="fro") ** 2
        B_sq_frob = np.linalg.norm(B, ord="fro") ** 2
        AB_nuc = np.linalg.norm(A.T @ B, ord="nuc")
        return A_sq_frob + B_sq_frob - 2 * AB_nuc

    def __call__(self, X, Y):
        return self.procrustes(X, Y)


class GPUProcrustes:
    """ Computes Procrustes distance between representations x and y
    GPU implementation based on https://github.com/js-d/sim_metric/blob/main/dists/scoring.py
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
        X_mean = tf.reduce_mean(X, axis=1, keepdims=True)
        X_centered = X - X_mean
        X_norm = X_centered / tf.norm(X_centered, ord="fro", axis=(0, 1))
        return X_norm

    def procrustes(self, X, Y):
        A = self.normalise(X)
        B = self.normalise(Y)
        logger.debug("Shape of x : {}, shape of y: {}".format(A.shape, B.shape))
        A_sq_frob = tf.norm(A, ord="fro", axis=(0, 1)) ** 2
        B_sq_frob = tf.norm(B, ord="fro", axis=(0, 1)) ** 2
        AB = tf.transpose(A) @ B
        AB_nuc = tf.reduce_sum(tf.linalg.svd(AB, compute_uv=False))
        return (A_sq_frob + B_sq_frob - 2 * AB_nuc).numpy()

    def __call__(self, X, Y):
        return self.procrustes(X, Y)
