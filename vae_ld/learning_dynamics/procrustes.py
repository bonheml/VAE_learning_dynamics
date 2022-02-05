import numpy as np
import tensorflow as tf
from vae_ld.learning_dynamics import logger
import jax.numpy as jnp


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
        X_norm /= np.linalg.norm(X_norm)
        return X_norm

    def procrustes(self, X, Y):
        A = self.center(X)
        B = self.center(Y)
        logger.debug("Shape of X : {}, shape of Y: {}".format(A.shape, B.shape))

        # In case dimensionality > nb_samples, we transpose A to get speedup the matrix multiplication done for the
        # Frobenius norm, taking advantage of the fact that the Frobenius norm of a matrix and its transpose are the same
        A_sq_frob = np.power(np.linalg.norm(A if A.shape[1] < A.shape[0] else A.T, ord="fro"), 2)
        B_sq_frob = np.power(np.linalg.norm(B if B.shape[1] < B.shape[0] else B.T, ord="fro"), 2)

        # In case both representations have the same shape dimensionality > nb_samples, we transpose B to speedup the
        # computation of the matrix multiplication. This will not impact the results of the nuclear norm and may speedup
        # the SVD.
        if A.shape == B.shape and A.shape[0] < A.shape[1]:
            AB = A @ B.T
        # Otherwise, we get AB of shape dim_A * dim_B
        else:
            AB = A.T @ B
        logger.debug("Shape of XTY : {}, dtype of XTY: {}".format(AB.shape, AB.dtype))

        AB_nuc = np.sum(jnp.linalg.svd(AB, compute_uv=False))
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



