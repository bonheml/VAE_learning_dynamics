import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from vae_ld.learning_dynamics import logger

class CKA:
    """
    Adapted from the demo code of "Similarity of Neural Network Representations Revisited", Kornblith et al. 2019
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
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
        m = tnp if self._use_gpu else np
        if self._debiased:
            # Unbiased version proposed by Szekely, G. J., & Rizzo, M. L. in
            # Partial distance correlation with methods for dissimilarities (2014) and implemented in
            # https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
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
