import numpy as np
import tensorflow as tf
from vae_ld.learning_dynamics import logger
import jax.numpy as jnp
from scipy.sparse.linalg import svds as sparse_svd


class Procrustes:
    """ Computes Procrustes distance between representations x and y
    Taken from https://github.com/js-d/sim_metric/blob/main/dists/scoring.py
    Implementation of Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
    """

    def __init__(self, name="procrustes", return_similarity=True, truncate_after=0):
        self._name = name
        self._return_similarity = return_similarity
        self._truncate_after = truncate_after

    @property
    def name(self):
        return self._name

    def center(self, X):
        # Here we use the same normalisation as in "Grounding Representation Similarity with Statistical Testing",
        # Ding et al. 2021, however, it gives inconsistent results.
        # X_norm = X - X.mean(axis=1, keepdims=True)
        # X_norm /= np.linalg.norm(X_norm)
        # Instead, we use the batch normalisation formula from "Batch Normalization: Accelerating Deep Network Training
        # by Reducing Internal Covariate Shift" Loffe et Szegedy, 2015 with offset of 0 and scale of 1:
        # (x - mu) / sigma
        mu, sigma = tf.nn.moments(X, keepdims=False)
        X_norm = tf.nn.batch_normalization(X, mu, sigma)

        return X_norm

    def procrustes(self, X, Y):
        logger.debug("Shape of X : {}, shape of Y: {}".format(X.shape, Y.shape))

        # In case dimensionality > nb_samples, we transpose A to get speedup the matrix multiplication done for the
        # Frobenius norm, taking advantage of the fact that the Frobenius norm of a matrix and its transpose are the same
        A_sq_frob = np.power(np.linalg.norm(X if X.shape[1] < X.shape[0] else X.T, ord="fro"), 2)
        B_sq_frob = np.power(np.linalg.norm(Y if Y.shape[1] < Y.shape[0] else Y.T, ord="fro"), 2)

        # In case both representations have the same shape dimensionality > nb_samples, we transpose B to speedup the
        # computation of the matrix multiplication. This will not impact the results of the nuclear norm and may speedup
        # the SVD.
        if X.shape == Y.shape and X.shape[0] < X.shape[1]:
            AB = X @ Y.T
        # Otherwise, we get AB of shape dim_A * dim_B
        else:
            AB = X.T @ Y
        logger.debug("Shape of XTY : {}, dtype of XTY: {}".format(AB.shape, AB.dtype))

        if self._truncate_after > 0:
            sigma = sparse_svd(AB, self._truncate_after, return_singular_vectors=False)
        else:
            sigma = jnp.linalg.svd(AB, compute_uv=False)

        AB_nuc = np.sum(sigma)
        # AB_nuc = np.linalg.norm(AB, ord="nuc")
        return A_sq_frob + B_sq_frob - 2 * AB_nuc

    def __call__(self, X, Y):
        procrustes_dist = self.procrustes(X, Y)
        return 1 - procrustes_dist / 2 if self._return_similarity else procrustes_dist


class GPUProcrustes:
    """ Computes Procrustes distance between representations x and y
    GPU implementation based on https://github.com/js-d/sim_metric/blob/main/dists/scoring.py
    Implementation of Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
    """

    def __init__(self, name="procrustes", return_similarity=True):
        self._name = name
        self._return_similarity = return_similarity

    @property
    def name(self):
        return self._name

    def center(self, X):
        # Here we use the same normalisation as in "Grounding Representation Similarity with Statistical Testing",
        # Ding et al. 2021, however, it gives inconsistent results
        # X_mean = tf.reduce_mean(X, axis=1, keepdims=True)
        # X_centered = X - X_mean
        # X_norm = X_centered / tf.norm(X_centered, ord="fro", axis=(0, 1))
        # Instead, we use the batch normalisation formula from "Batch Normalization: Accelerating Deep Network Training
        # by Reducing Internal Covariate Shift" Loffe et Szegedy, 2015 with offset of 0 and scale of 1:
        # (x - mu) / sigma

        mu, sigma = tf.nn.moments(X, keepdims=False)
        X_norm = tf.nn.batch_normalization(X, mu, sigma)
        return X_norm

    def procrustes(self, X, Y):
        logger.debug("Shape of x : {}, shape of y: {}".format(X.shape, Y.shape))
        # In case dimensionality > nb_samples, we transpose A to get speedup the matrix multiplication done for the
        # Frobenius norm, taking advantage of the fact that the Frobenius norm of a matrix and its transpose are the same
        A_sq_frob = tf.norm(X if X.shape[1] < X.shape[0] else tf.transpose(X), ord="fro", axis=(0, 1)) ** 2
        B_sq_frob = tf.norm(Y if Y.shape[1] < Y.shape[0] else tf.transpose(Y), ord="fro", axis=(0, 1)) ** 2

        # In case both representations have the same shape dimensionality > nb_samples, we transpose B to speedup the
        # computation of the matrix multiplication. This will not impact the results of the nuclear norm and may speedup
        # the SVD.
        if X.shape == Y.shape and X.shape[0] < X.shape[1]:
            AB = X @ tf.transpose(Y)
        # Otherwise, we get AB of shape dim_A * dim_B
        else:
            AB = tf.transpose(X) @ Y

        AB_nuc = tf.reduce_sum(tf.linalg.svd(AB, compute_uv=False))
        return (A_sq_frob + B_sq_frob - 2 * AB_nuc).numpy()

    def __call__(self, X, Y):
        procrustes_dist = self.procrustes(X, Y)
        return 1 - procrustes_dist / 2 if self._return_similarity else procrustes_dist



