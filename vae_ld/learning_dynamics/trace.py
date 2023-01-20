import numpy as np
from vae_ld.learning_dynamics import logger


class CovarianceTrace:
    def __init__(self, precision=None):
        """
        Parameters
        ----------
        precision : int or None
            The precision to which the covariance matrix is rounded
        """
        self._precision = precision

    def fit_transform(self, X):
        """Computes the trace of the covariance of a representation

        Parameters
        ----------
        X : np.array
            A (n_example, n_features) matrix

        Returns
        -------
        float
            The trace of cov[X]
        """
        cov = np.cov(X, rowvar=False)
        if self._precision:
            cov = cov.round(decimals=self._precision)
        logger.debug("Cov[X] = {}".format(cov))
        tr = np.trace(cov)
        logger.debug("Tr(Cov[X]) = {}".format(tr))
        return tr