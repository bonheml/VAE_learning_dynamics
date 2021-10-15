from scipy.spatial.distance import squareform, pdist
import numpy as np

from vae_ld.learning_dynamics import logger


class TwoNN:
    """ Implementation of the ID estimator TwoNN from [1]

    [1] Estimating the intrinsic dimension of datasets by a minimal neighborhood information
        Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio, 2017
    """
    def __init__(self):
        self._to_keep = 0.9

    @property
    def to_keep(self):
        return self._to_keep

    @to_keep.setter
    def to_keep(self, to_keep):
        """ Set the fraction of data points to keep during the ID estimate
        """
        if to_keep <= 0 or to_keep > 1:
            raise ValueError("The fraction to keep must be between 0 (excluded) and 1.")
        self._to_keep = to_keep

    def get_id_estimate(self, X):
        """ Compute the intrinsic dimension estimation, based on the implementation of [1] and [2].
        The steps described in [3] (p.3) are outlined in the code comments.

        [1] https://github.com/efacco/TWO-NN (C++ implementation by the authors of [3])
        [2] https://github.com/ansuini/IntrinsicDimDeep (Python implementation by the authors of [4])
        [3] Estimating the intrinsic dimension of datasets by a minimal neighborhood information
            Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio, 2017
        [4] Intrinsic dimension of data representations in deep neural networks
            Alessio Ansuini, Alessandro Laio, Jakob H. Macke, and Davide Zoccolan, 2019
        """
        # 1. Compute the pairwise distances for each point in the dataset
        logger.info("Computing the pairwise distance between each point of the dataset")
        x_dist = np.sort(squareform(pdist(X)), axis=1, kind="heapsort")

        # 2. Get two shortest distances
        logger.info("Getting the two shortest distances")
        r1 = x_dist[:, 1]
        r2 = x_dist[:, 2]

        logger.info("Removing zero values and degeneracies")
        zeros = np.where(r1 == 0)[0]
        degeneracies = np.where(r1 == r2)[0]
        good = np.setdiff1d(np.arange(x_dist.shape[0]), np.array(zeros))
        good = np.setdiff1d(good, np.array(degeneracies))
        r1 = r1[good]
        r2 = r2[good]

        # 3. For each point i compute mu_i
        logger.info("Computing mu_i for each point i")
        mu = np.sort(r2/r1, kind="heapsort")

        # 4. Compute the empirical cumulate Femp(mu)
        logger.info("Computing the empirical cumulate")
        n = r1.shape[0]
        Femp = np.arange(0, n, dtype=np.float64) / n

        # 5. Fit the points of the plane given by coordinates {(log(mu_i), -log(1 - Femp(mu_i)))|i=1, …, n} with a
        # straight line passing through the origin, using the analytical solution of the linear regression.
        # Note that we discard 10% of the points by default, as recommended in the TwoNN paper
        logger.info("Fitting the {}% first points with a linear regression".format(self._to_keep * 100))
        n_to_keep = int(n * self._to_keep)
        x = np.log(mu)[:n_to_keep]
        y = -np.log(1 - Femp)[:n_to_keep]
        d = np.dot(x, y) / np.dot(x, x)

        return d
