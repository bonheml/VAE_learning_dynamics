import numpy as np
import tensorflow as tf
from vae_ld.learning_dynamics import logger
import tensorflow.experimental.numpy as tnp

class Procrustes:
    """ Computes Procrustes distance between representations x and y
    Taken from https://github.com/js-d/sim_metric/blob/main/dists/scoring.py
    Implementation of Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
    """

    def __init__(self, name="procrustes", use_gpu=False):
        self._name = name
        self._gpu = use_gpu

    @property
    def name(self):
        return self._name

    def center(self, X):
        m = tnp if self._gpu else np
        # Here when self.normalised is True, we use the same normalisation as in
        # "Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
        X_norm = X - m.mean(X, axis=0, keepdims=True)
        X_norm /= m.linalg.norm(X_norm)

        return X_norm

    def procrustes(self, X, Y):
        m = tnp if self._gpu else np
        logger.debug("Shape of X : {}, shape of Y: {}".format(X.shape, Y.shape))
        A_sq_frob = m.sum(X ** 2)
        B_sq_frob = m.sum(Y ** 2)

        AB = m.transpose(X) @ Y
        logger.debug("Shape of XTY : {}, dtype of XTY: {}".format(AB.shape, AB.dtype))
        if self._gpu:
            AB_nuc = tf.reduce_sum(tf.linalg.svd(AB, compute_uv=False))
        else:
            AB_nuc = m.linalg.norm(AB, ord="nuc")
        return A_sq_frob + B_sq_frob - 2 * AB_nuc

    def __call__(self, X, Y):
        procrustes_dist = self.procrustes(X, Y)
        return 1 - procrustes_dist / 2
