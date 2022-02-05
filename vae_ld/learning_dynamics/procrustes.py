import numpy as np
import tensorflow as tf
from vae_ld.learning_dynamics import logger
from scipy.linalg.interpolative import svd


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

    def center(self, X):
        # Here we use the same normalisation as in "Grounding Representation Similarity with Statistical Testing",
        # Ding et al. 2021
        X_norm = X - X.mean(axis=1, keepdims=True)
        X_norm /= np.linalg.norm(X_norm).astype("float64")
        return X_norm

    def procrustes(self, X, Y):
        A = self.center(X)
        B = self.center(Y)
        logger.debug("Shape of x : {}, shape of y: {}".format(A.shape, B.shape))
        A_sq_frob = np.linalg.norm(A, ord="fro") ** 2
        B_sq_frob = np.linalg.norm(B, ord="fro") ** 2
        AB = A.T @ B
        logger.debug("Shape of XTY : {}, dtype of XTY: {}".format(AB.shape, AB.dtype))
        # Compute interpolative SVD with relative error < 0.01 to make the computation possible on large convolutional
        # layers
        AB_nuc = np.sum(svd(AB.astype("float64"), 0.01)[1])
        # AB_nuc = np.linalg.norm(AB, ord="nuc")
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

    def center(self, X):
        # Here we use the same normalisation as in "Grounding Representation Similarity with Statistical Testing",
        # Ding et al. 2021
        X_mean = tf.reduce_mean(X, axis=1, keepdims=True)
        X_centered = X - X_mean
        X_norm = X_centered / tf.norm(X_centered, ord="fro", axis=(0, 1))
        return X_norm

    def procrustes(self, X, Y):
        logger.debug("Shape of x : {}, shape of y: {}".format(A.shape, B.shape))
        A_sq_frob = tf.norm(X, ord="fro", axis=(0, 1)) ** 2
        B_sq_frob = tf.norm(Y, ord="fro", axis=(0, 1)) ** 2
        AB = tf.transpose(X) @ Y
        AB_nuc = tf.reduce_sum(tf.linalg.svd(AB, compute_uv=False))
        return (A_sq_frob + B_sq_frob - 2 * AB_nuc).numpy()

    def __call__(self, X, Y):
        return self.procrustes(X, Y)



