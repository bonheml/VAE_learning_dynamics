import random

from scipy.spatial.distance import squareform, pdist
import numpy as np
from sklearn import linear_model
import gibbs
from sklearn.neighbors import NearestNeighbors
from vae_ld.learning_dynamics import logger


class TwoNN:
    """ Implementation of the ID estimator TwoNN from [1]

    [1] Estimating the intrinsic dimension of datasets by a minimal neighborhood information
        Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio, 2017
    """
    def __init__(self):
        self._to_keep = 0.9
        self._knn = NearestNeighbors(n_neighbors=3)

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

    def fit_transform(self, X):
        """ Compute the intrinsic dimension estimation, based on the implementation of [1] and [2].
        The steps described in [3] (p.3) are outlined in the code comments.

        [1] https://github.com/efacco/TWO-NN (C++ implementation by the authors of [3])
        [2] https://github.com/ansuini/IntrinsicDimDeep (Python implementation by the authors of [4])
        [3] Estimating the intrinsic dimension of datasets by a minimal neighborhood information
            Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio, 2017
        [4] Intrinsic dimension of data representations in deep neural networks
            Alessio Ansuini, Alessandro Laio, Jakob H. Macke, and Davide Zoccolan, 2019
        """
        self._knn.fit(X)
        # 1. Compute the pairwise distances for each point in the dataset
        logger.info("Computing the pairwise distance between each point of the dataset")
        # x_dist = np.sort(squareform(pdist(X)), axis=1, kind="heapsort")
        x_dist = self._knn.kneighbors(X)[0]

        # 2. Get two shortest distances
        logger.info("Getting the two shortest distances")
        r1 = x_dist[:, 1]
        r2 = x_dist[:, 2]

        # This step was added in Ansuini et al. implementation
        # logger.info("Removing zero values and degeneracies")
        # zeros = np.where(r1 == 0)[0]
        # degeneracies = np.where(r1 == r2)[0]
        # good = np.setdiff1d(np.arange(x_dist.shape[0]), np.array(zeros))
        # good = np.setdiff1d(good, np.array(degeneracies))
        # logger.info(good.shape)
        # r1 = r1[good]
        # r2 = r2[good]

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


class MLE:
    def __init__(self, k, seed, runs=5, anchor=0.9):
        self._anchor = anchor
        self._k = k
        self._seed = seed
        self._n_runs = runs
        self._knn = NearestNeighbors(n_neighbors=k+1)

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        """ Set the fraction of data points to keep during the ID estimate
        """
        if anchor <= 0 or anchor > 1:
            raise ValueError("The anchor fraction must be between 0 (excluded) and 1.")
        self._anchor = anchor

    @property
    def k(self):
        return self._k

    @k.setter
    def anchor(self, k):
        """ Set the fraction of data points to keep during the ID estimate
        """
        if k <= 0:
            raise ValueError("The number of neighbours must be greater than 0.")
        self._k = k

    def fit_transform(self, X):
        anchor_samples = int(self.anchor * X.shape[0])
        res = np.zeros((self._n_runs,))
        data_idxs = np.arange(X.shape[0])
        self._knn.fit(X)

        for i in range(self._n_runs):
            logger.info("Computing iteration {} of MLE with k={}".format(i, self._k))
            np.random.shuffle(data_idxs)
            anchor_idxs = data_idxs[:anchor_samples]
            res[i] = self._compute_mle(X[anchor_idxs])

        return res.mean()

    def _compute_mle(self, X):
        dist = self._knn.kneighbors(X)[0][:, 1:]

        if not np.all(dist > 0.):
            logger.info(np.argwhere(dist <= 0.))
            logger.info(dist[np.argwhere(dist <= 0.)])

        assert np.all(dist > 0.)

        d = np.log(dist[:, self._k - 1: self._k] / dist[:, 0:self._k - 1])
        d = d.sum(axis=1) / (self.k - 2)

        return 1. / d.mean()


class Hidalgo:
    """ Compute Hidalgo, an algorithm initially proposed in [1].
    The implementation is from https://github.com/micheleallegra/Hidalgo/tree/master/python,
    the code released with [1].

    [1] Data segmentation based on the local intrinsic dimension, Allegra et al., 2020
    """
    def __init__(self, metric='euclidean', k=2, zeta=0.8, q=3, iters=10000, replicas=10, burn_in=0.9):
        """
        :param metric: The metric to use for KNN, if predefined, then a distance matrix will be given when calling fit
        :param k: The number of manifolds
        :param zeta: The probability to sample the neighbour of a point from the same manifold (in the paper's formula,
        this is xsi)
        :param q: number of closest neighbours from each points to keep
        :param iters: number of iterations of the Gibbs sampling
        :param replicas: number of times the sampling should be replicated
        :param burn_in: percentage of points to exclude of the estimation
        """
        self.metric = metric
        self.k = k
        self.zeta = zeta
        self.q = q
        self.iters = iters
        self.burn_in = burn_in
        self.replicas = replicas

        # Setting prior parameters of d to 1
        self.a = np.ones(k)
        self.b = np.ones(k)

        # Setting prior parameter of p to 1
        self.c = np.ones(k)

        # Setting prior parameter of zeta to 1
        self.f = np.ones(k)

        # Setting the save samples every 10 sampling and compute the total number of samples
        self.sampling_rate = 10
        self.n_samples = np.floor((self.iters - np.ceil(self.burn_in * self.iters)) / self.sampling_rate).astype(int)

        # z will not be fixed
        self.fixed_z = 0

        # Local interaction between z are used
        self.use_local_z_interaction = 1

        # z will not be updated during the training
        self.update_z = 0

    def _fit(self, X):
        assert isinstance(X, np.ndarray), "X should be a numpy array"
        assert len(np.shape(X)) == 2, "X should be a two-dimensional numpy array"
        n, d = np.shape(X)
        nns_mat = np.zeros((n, n))

        logger.info("Getting the {} nearest neighbours from each point".format(self.q))
        if self.metric == "predefined":
            distances = np.sort(X)[:, :self.q + 1]
            indices_in = np.argsort(X)[:, :self.q + 1]
        else:
            nns = NearestNeighbors(n_neighbors=self.q + 1, algorithm="ball_tree", metric=self.metric).fit(X)
            distances, indices_in = nns.kneighbors(X)

        for i in range(self.q):
            nns_mat[indices_in[:, 0], indices_in[:, i + 1]] = 1

        nns_count = np.sum(nns_mat, axis=0)
        indices_out = np.where(nns_mat.T)[1]
        indices_track = np.cumsum(nns_count)
        indices_track = np.append(0, indices_track[:-1])
        mu = np.divide(distances[:, 2], distances[:, 1])
        n_par = n + 2 * self.k + 2
        best_sampling = np.zeros((self.n_samples, n_par))
        indices_in = indices_in[:, 1:]
        indices_in = np.reshape(indices_in, (n * self.q,))

        threshold = -1.E10
        for i in range(self.replicas):
            logger.info("Doing Gibbs sampling {}/{}".format(i + 1, self.replicas))
            sampling = 2 * np.ones(self.n_samples * n_par)
            gibbs.GibbsSampling(self.iters, self.k, self.fixed_z, self.use_local_z_interaction, self.update_z, self.q,
                                self.zeta, self.sampling_rate, self.burn_in, i, mu, indices_in.astype(float),
                                indices_out.astype(float), nns_count, indices_track, self.a, self.b, self.c, self.f,
                                sampling)
            sampling = np.reshape(sampling, (self.n_samples, n_par))
            lik = np.mean(sampling[:, -1], axis=0)
            if lik > threshold:
                logger.info("Better likelihood obtained with replica {}".format(i + 1))
                best_sampling = sampling
                threshold = lik

        return best_sampling, self.n_samples

    def fit(self, X):
        n = np.shape(X)[0]
        sampling, n_samples = self._fit(X)
        p_i = np.zeros((self.k, n))

        for i in range(self.k):
            p_i[i, :] = np.sum(sampling[:, 2 * self.k:2 * self.k + n] == i, axis=0)

        z = np.argmax(p_i, axis=0)
        p_z = np.max(p_i, axis=0)
        z = z + 1
        z[np.where(p_z < 0.8)] = 0

        res = dict()
        res["k"] = self.k
        res["samples"] = n_samples
        res["z"] = z.tolist()
        res["p_i"] = (p_i / n_samples).tolist()
        res["d"] = np.mean(sampling[:, :self.k], axis=0).tolist()
        res["d_err"] = np.std(sampling[:, :self.k], axis=0).tolist()
        res["p"] = np.mean(sampling[:, self.k:2 * self.k], axis=0).tolist()
        res["p_err"] = np.std(sampling[:, self.k:2 * self.k], axis=0).tolist()
        res["likelihood"] = np.mean(sampling[:, -1], axis=0).tolist()
        res["likelihood_err"] = np.std(sampling[:, -1], axis=0).tolist()

        return res
