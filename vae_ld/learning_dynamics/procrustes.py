import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from scipy import linalg

from vae_ld.learning_dynamics import logger


class Procrustes:
    """ Computes Procrustes distance between representations x and y based on Ding et al [1]
    `implementation <https://github.com/js-d/sim_metric/blob/main/dists/scoring.py>`_.

    Parameters
    ----------
    name : str, optional
        The name of the metric. Default "procrustes".
    use_gpu : bool, optional
        If True, use tensorflow implementation else, use numpy implementation. Default False.

    References
    ----------
    .. [1] Ding, F., Denain, J. S., & Steinhardt, J. (2021). Grounding Representation Similarity with Statistical
           Testing. arXiv preprint arXiv:2108.01661.
    """

    def __init__(self, name="procrustes", use_gpu=False):
        """

        Parameters
        ----------
        name
        use_gpu
        """
        self._name = name
        self._gpu = use_gpu

    @property
    def name(self):
        return self._name

    def center(self, X):
        """ Normalise `X` so that its mean is 0 and its Frobenius norm is 1.

        Parameters
        ----------
        X : np.array
            The matrix to normalise

        Returns
        -------
        np.array
            The normalised matrix.
        """
        # Here when self.normalised is True, we use the same normalisation as in
        # "Grounding Representation Similarity with Statistical Testing", Ding et al. 2021
        if self._gpu:
            X_norm = X - tnp.mean(X, axis=0, keepdims=True)
            X_norm /= tf.norm(X_norm)
        else:
            X = np.asfortranarray(X, dtype=np.float32)
            X_norm = X - np.mean(X, axis=0, keepdims=True)
            X_norm /= np.linalg.norm(X_norm)

        return X_norm

    def procrustes(self, X, Y):
        """ Compute The Procrustes similarity between `X` and `Y`. Both matrices are assumed to be normalised and to
        contain the same number of data examples `n`.

        Parameters
        ----------
        X : np.array
            A centered matrix of size (n, m)
        Y : np.array
            A centered matrix of size (n, p)

        Returns
        -------
        float
            A Procrustes similarity between 0 (not similar at all) and 1 (identical).

        """
        m = tnp if self._gpu else np
        logger.debug("Shape of X : {}, shape of Y: {}".format(X.shape, Y.shape))
        A_sq_frob = m.sum(X ** 2)
        B_sq_frob = m.sum(Y ** 2)

        if self._gpu:
            AB = m.transpose(X) @ Y
            logger.debug("Shape of XTY : {}, dtype of XTY: {}".format(AB.shape, AB.dtype))
            AB_nuc = tf.reduce_sum(tf.linalg.svd(AB, compute_uv=False))
        else:
            # Speedup matrix multiplication for large matrices
            AB = linalg.blas.sgemm(1.0, X.T, Y)
            logger.debug("Shape of XTY : {}, dtype of XTY: {}".format(AB.shape, AB.dtype))
            # Speedup nuclear norm for large matrices
            AB_nuc = np.sum(linalg.svd(AB, compute_uv=False, overwrite_a=True, check_finite=False))
        return 1 - (A_sq_frob + B_sq_frob - 2 * AB_nuc) / 2

    def __call__(self, X, Y):
        return self.procrustes(X, Y)

