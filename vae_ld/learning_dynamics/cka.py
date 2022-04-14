import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from vae_ld.learning_dynamics import logger

class CKA:
    """ Compute the linear Centered Kernel Alignment (CKA).
    Adapted from the `demo code <https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ>`_
    of Kornblith et al. [1].

    Parameters
    ----------
    name : str, optional
        The name of the metric. Default "CKA".
    debiased : bool, optional
        If True, use the debiased implementation of CKA proposed by Székely et al. [2] instead of the standard one. Default False.
    use_gpu : bool, optional
        If True, use tensorflow implementation else, use numpy implementation. Default False.

    References
    ----------
    .. [1] Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019, May). Similarity of neural network representations
           revisited. In International Conference on Machine Learning (pp. 3519-3529). PMLR.
    .. [2] Székely, G. J., & Rizzo, M. L. (2014). Partial distance correlation with methods for dissimilarities.
           The Annals of Statistics, 42(6), 2382-2412.
    """
    def __init__(self, name="CKA", debiased=False, use_gpu=False):
        self._debiased = debiased
        self._name = name
        self._use_gpu = use_gpu

    @property
    def debiased(self):
        return self._debiased

    @debiased.setter
    def debiased(self, debiased):
        self._debiased = debiased

    @property
    def name(self):
        return self._name

    def center(self, x):
        """ Center the matrix `x` so that its mean is 0.

        Parameters
        ----------
        x : np.array
            The matrix to center

        Returns
        -------
        np.array
            The centered matrix.
        """
        m = tnp if self._use_gpu else np
        if self._debiased:
            x = m.dot(x, x.T)
            n = x.shape[0]
            m.fill_diagonal(x, 0)
            means = m.sum(x, 0, dtype=m.float64) / (n - 2)
            means -= m.sum(means) / (2 * (n - 1))
            x_centred = x - means[None, :]
            x_centred -= means[:, None]
            m.fill_diagonal(x_centred, 0)
        else:
            x_centred = x - m.mean(x, axis=0, keepdims=True)
            x_centred = m.dot(x_centred, m.transpose(x_centred))
        return x_centred

    def cka(self, x, y):
        """ Compute The CKA score between `x` and `y`. Both matrices are assumed to be centered and to contain the same
        number of data examples `n`.

        Parameters
        ----------
        x : np.array
            A centered matrix of size (n, n)
        y : np.array
            A centered matrix of size (n, n)

        Returns
        -------
        float
            A CKA score between 0 (not similar at all) and 1 (identical).
        """
        m = tnp if self._use_gpu else np
        # Note: this method assumes that kc and lc are the centered kernel values given by cka.center(cka.kernel(.))
        # Compute tr(KcLc) = vec(kc)^T vec(lc), omitting the term (m-1)**2, which is canceled by CKA
        hsic = m.dot(m.ravel(x), m.ravel(y))
        if self._use_gpu:
            normalization_x = tf.norm(x)
            normalization_y = tf.norm(y)
        else:
            normalization_x = m.linalg.norm(x)
            normalization_y = m.linalg.norm(y)
        cka = hsic / (normalization_x * normalization_y)
        # Divide by the product of the Frobenius norms of kc and lc to get CKA
        logger.debug("CKA = {} / ({} * {}) = {}".format(hsic, normalization_x, normalization_y, cka))
        return cka

    def __call__(self, x, y):
        return self.cka(x, y)
