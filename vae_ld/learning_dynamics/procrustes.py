import numpy as np
import tensorflow as tf
from vae_ld.learning_dynamics import logger


class Procrustes:
    """ Computes Procrustes distance between representations x and y
    Taken from https://github.com/js-d/sim_metric/blob/main/dists/scoring.py
    Implementation of Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
    """

    def __init__(self, name="procrustes", use_gpu=False, normalised=False):
        self._name = name
        self._normalised = normalised
        self._gpu = use_gpu

    @property
    def name(self):
        return self._name

    def _center_gpu(self, X):
        # Here when self.normalised is True, we use the same normalisation as in
        # "Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
        X_mean = tf.reduce_mean(X, axis=1, keepdims=True)
        X_norm = X - X_mean

        if self._normalised:
            X_norm /= tf.norm(X_norm, ord="fro")

        return X_norm

    def center(self, X):
        if self._gpu:
            return self._center_gpu(X)
        # Here when self.normalised is True, we use the same normalisation as in
        # "Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
        X_norm = X - X.mean(axis=1, keepdims=True)

        if self._normalised:
            X_norm /= np.linalg.norm(X_norm, ord="fro")

        return X_norm

    def procrustes(self, X, Y):
        logger.debug("Shape of X : {}, shape of Y: {}".format(X.shape, Y.shape))
        if self._gpu:
            return self._procrustes_gpu(X, Y)

        if self._normalised:
            # Following the implementation of "Grounding Representation Similarity with Statistical Testing",
            # Ding et al. 2021
            A_sq_frob = np.sum(X ** 2)
            B_sq_frob = np.sum(Y ** 2)
        else:
            A_sq_frob = np.linalg.norm(X, ord="fro") ** 2
            B_sq_frob = np.linalg.norm(Y, ord="fro") ** 2

        AB = X.T @ Y
        logger.debug("Shape of XTY : {}, dtype of XTY: {}".format(AB.shape, AB.dtype))
        AB_nuc = np.linalg.norm(AB, ord="nuc")
        return A_sq_frob + B_sq_frob - 2 * AB_nuc

    def _procrustes_gpu(self, X, Y):
        if self._normalised:
            # Following the implementation of "Grounding Representation Similarity with Statistical Testing",
            # Ding et al. 2021
            A_sq_frob = tf.reduce_sum(X ** 2)
            B_sq_frob = tf.reduce_sum(Y ** 2)
        else:
            A_sq_frob = tf.norm(X, ord="fro") ** 2
            B_sq_frob = tf.norm(Y, ord="fro") ** 2

        # Compute the nuclear norm (i.e., sum of AB's singular values)
        AB_nuc = tf.reduce_sum(tf.linalg.svd(tf.transpose(X) @ Y, compute_uv=False))
        return (A_sq_frob + B_sq_frob - 2 * AB_nuc).numpy()

    def __call__(self, X, Y):
        procrustes_dist = self.procrustes(X, Y)
        return 1 - procrustes_dist / 2 if self._normalised else procrustes_dist
