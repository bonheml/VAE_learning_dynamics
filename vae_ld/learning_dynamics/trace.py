import numpy as np
from vae_ld.learning_dynamics import logger


class CovarianceTrace:

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
        logger.debug("Cov[X] = {}".format(cov))
        tr = np.trace(cov)
        logger.debug("Tr(Cov[X]) = {}".format(tr))
        return tr