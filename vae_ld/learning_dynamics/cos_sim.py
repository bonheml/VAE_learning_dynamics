import math
import numpy as np
import tensorflow as tf
from sklearn import metrics
from vae_ld.learning_dynamics import logger


class CosineSim:
    """ Computes the cosine similarity between representations x and y.

    Parameters
    ----------
    name : str, optional
        The name of the metric. Default "cos_sim".
    use_gpu : bool, optional
        If True, use tensorflow implementation else, use numpy implementation. Default False.
    agg: bool, optional
        If True, sum the cosine similarity obtained for each feature, else return a vector of cosine similarities.
        Default True.
    squared: bool, optional
        If True, return the squared cosine similarity. Default False.
    """
    def __init__(self, name="cos_sim", use_gpu=False, agg=True, squared=False):
        self._name = name
        self._gpu = use_gpu
        self._agg = agg
        self._squared = squared

    @property
    def name(self):
        return self._name

    def center(self, X):
        return X

    def cos_sim(self, X, Y):
        """ Compute the cosine similarity between `X` and `Y`. Both matrices are assumed to
        contain the same number of data examples `n` and the same number of features `m`.

        Parameters
        ----------
        X : np.array
            A matrix of size (n, m)
        Y : np.array
            A matrix of size (n, m)

        Returns
        -------
        float
            A cosine similarity between -1 and 1.
            The further away from 0 the score is, the more similar the values are.
        """
        logger.debug("Shape of X : {}, shape of Y: {}".format(X.shape, Y.shape))

        if self._gpu:
            res = -tf.keras.losses.cosine_similarity(X, Y, axis=0)
            logger.info("Cosine similarity is {}".format(res))
            if self._squared is True:
                res = tf.square(res)
                logger.info("Squared cosine similarity is {}".format(res))
            if self._agg is True:
                res = tf.reduce_sum(res)
                logger.info("Summed cosine similarity is {}".format(res))
            res = res.numpy()
        else:
            res = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                res[i] = metrics.pairwise.cosine_similarity(X[:, i].reshape(1, -1), Y[:, i].reshape(1, -1))
            logger.info("Cosine similarity is {}".format(res))
            if self._squared is True:
                res = np.square(res)
                logger.info("Squared cosine similarity is {}".format(res))
            if self._agg is True:
                res = np.sum(res)
                logger.info("Summed cosine similarity is {}".format(res))
        return res

    def __call__(self, X, Y):
        return self.cos_sim(X, Y)


class AngularSim:
    """ Computes the angular similarity between representations x and y.

    Parameters
    ----------
    name : str, optional
        The name of the metric. Default "ang_sim".
    use_gpu : bool, optional
        If True, use tensorflow implementation else, use numpy implementation. Default False.
    agg: bool, optional
        If True, sum the angular similarity obtained for each feature, else return a vector of angular similarities.
        Default True.
    """

    def __init__(self, name="ang_sim", use_gpu=False, agg=True):
        self._name = name
        self._gpu = use_gpu
        self._agg = agg
        self._cos_sim = CosineSim(use_gpu=self._gpu, agg=False)

    @property
    def name(self):
        return self._name

    def center(self, X):
        return X

    def ang_sim(self, X, Y):
        """ Compute the angular similarity between `X` and `Y`. Both matrices are assumed to
        contain the same number of data examples `n` and the same number of features `m`.

        Parameters
        ----------
        X : np.array
            A matrix of size (n, m)
        Y : np.array
            A matrix of size (n, m)

        Returns
        -------
        float
            An angular similarity between 0 (not similar) and 1 (identical).
        """
        logger.debug("Shape of X : {}, shape of Y: {}".format(X.shape, Y.shape))
        sim = self._cos_sim(X, Y)

        if self._gpu:
            res = 1 - tf.acos(sim) / math.pi
            logger.info("Angular similarity is {}".format(res))
            if self._agg is True:
                res = tf.reduce_sum(res)
                logger.info("Summed angular similarity is {}".format(res))
            res = res.numpy()
        else:
            res = 1 - np.arccos(sim) / math.pi
            logger.info("Angular similarity is {}".format(res))
            if self._agg is True:
                res = np.sum(res)
                logger.info("Summed angular similarity is {}".format(res))
        return res

    def __call__(self, X, Y):
        return self.ang_sim(X, Y)