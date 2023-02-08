import numpy as np
from vae_ld.learning_dynamics.utils import truncate
from vae_ld.learning_dynamics import logger


def gram_rbf(X):
    """ Gram of RBF kernel for MI evaluation of bottleneck size in [1].
    Reuse the implementation of [2] with sigma defined based on Silverman's rule, as in [1]
    Parameters
    ----------
    X: np.array
        The data to transform

    Returns
    -------
    np.array
        The kernelised data

    [1] Boquet, Guillem, et al. "Theoretical Tuning of the Autoencoder Bottleneck Layer Dimension: A Mutual
    Information-based Algorithm." 2020 28th European Signal Processing Conference (EUSIPCO). IEEE, 2021.
    [2] Kornblith, Simon, et al. "Similarity of neural network representations revisited." International Conference on
    Machine Learning. PMLR, 2019.
    """
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    sigma = np.std(X) * X.shape[0] ** (-1 / (4 + X.shape[1]))
    dot_products = X.dot(X.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    return np.exp(-sq_distances / (2 * sigma ** 2))


class InformationBottleneck:
    def __init__(self, batch_size=100, precision=2, alpha=2):
        self.precision = precision
        self.alpha = alpha
        self.batch_size = batch_size

    def fit_transform(self, X, X2=None):
        res = []
        for i in range(0, X.shape[0], self.batch_size):
            if X2 is not None:
                res.append(truncate(self.get_mutual_info(X[i:i + self.batch_size - 1], X2[i:i + self.batch_size - 1]),
                                    self.precision))
            else:
                res.append(truncate(self.get_mutual_info(X[i:i + self.batch_size - 1]), self.precision))
        return np.mean(res)

    def get_s_alpha(self, A):
        eig_sum = np.sum(np.linalg.eigvalsh(A) ** self.alpha)
        return 1 / (1 - self.alpha) * np.log2(eig_sum)

    def get_mutual_info(self, X, X2=None):
        A = gram_rbf(X)
        A /= np.trace(A)
        sa = self.get_s_alpha(A)
        if X2 is None:
            return sa
        B = gram_rbf(X2)
        B /= np.trace(B)
        sb = self.get_s_alpha(B)
        schur_ab = A * B
        sab = self.get_s_alpha(schur_ab / np.trace(schur_ab))
        return sa + sb - sab
