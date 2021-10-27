import numpy as np
import tensorflow as tf
import pandas as pd


def linear_kernel(x):
    return x.dot(x.T)


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

        x = x.copy()
        if self.debiased:
            # Unbiased version proposed by Szekely, G. J., & Rizzo, M. L. in
            # Partial distance correlation with methods for dissimilarities (2014) and implemented in
            # https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
            n = x.shape[0]
            np.fill_diagonal(x, 0)
            means = np.sum(x, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            x -= means[None, :]
            x -= means[:, None]
            np.fill_diagonal(x, 0)
        else:
            means = np.mean(x, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            x -= means[:, None]
            x -= means[None, :]
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


def get_activations(data, model_path):
    model = tf.keras.models.load_model(model_path)
    acts = model.encoder(data, training=False)
    acts += model.decoder(acts[-1], training=False)
    # Note that one could get weights using l.get_weights() instead of l.name here
    layer_names = [l.name for l in model.encoder.layers]
    layer_names += [l.name for l in model.decoder.layers]
    return model, acts, layer_names


def compute_models_cka(cka, data, m1_path, m2_path, save_path):
    m1, acts1, layers1 = get_activations(data, m1_path)
    m2, acts2, layers2 = get_activations(data, m2_path)
    res = {}
    for i, l1 in enumerate(layers1):
        x = acts1[i]
        res[l1] = {}
        for j, l2 in enumerate(layers2):
            y = acts2[j]
            res[l1][l2] = cka(x, y)
    res = pd.DataFrame(res).T
    res.to_csv(save_path, sep="\t")

