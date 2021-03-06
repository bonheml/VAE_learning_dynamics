import numpy as np


def linear_kernel(x):
    return np.dot(x, x.T)


def rbf_kernel(x, threshold=1.0):
    """
    Taken from the demo code of "Similarity of Neural Network Representations Revisited", Kornblith et al. 2019
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ

    :param x: The matrix of features with size (num_examples x num_features)
    :param threshold: The fraction of the median Euclidian distance to use.
    :return: The transformed matrix of size (num_examples x num_examples)
    """
    xxt = np.dot(x, x.T)
    sq_norms = np.diag(xxt)
    sq_distances = -2 * xxt + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


class CKA:
    """
    Adapted from the demo code of "Similarity of Neural Network Representations Revisited", Kornblith et al. 2019
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
    """
    def __init__(self, kernel=linear_kernel, debiased=False):
        self.kernel = kernel
        self.debiased = debiased

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        self._kernel = kernel

    @property
    def debiased(self):
        return self._debiased

    @debiased.setter
    def debiased(self, debiased):
        self._debiased = debiased

    def center(self, x):
        n = x.shape[0]
        if self.debiased:
            # Unbiased version proposed by Szekely, G. J., & Rizzo, M. L. in
            # Partial distance correlation with methods for dissimilarities (2014).
            # and implemented in https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
            np.fill_diagonal(x, 0)
            means = np.sum(x, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            x = x - means[None, :] - means[:, None]
            np.fill_diagonal(x, 0)
        else:
            means = np.sum(x, 0, dtype=np.float64) / n
            means -= np.sum(means) / (2 * n)
            x = x - means[None, :] - means[:, None]
        return x

    def cka(self, x, y):
        kc = self.center(self.kernel(x))
        lc = self.center(self.kernel(y))
        # Compute tr(KcLc) = vec(kc)^T vec(lc), omitting the term (m-1)**2, which is canceled by CKA
        hsic = np.dot(kc.ravel(), lc.ravel())
        # Divide by the product of the Frobenius norms of kc and lc to get CKA
        return hsic / (np.linalg.norm(kc, ord="fro") * np.linalg.norm(lc, ord="fro"))

    def __call__(self, x, y):
        return self.cka(x, y)
