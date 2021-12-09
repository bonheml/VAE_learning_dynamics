import numpy as np
import tensorflow as tf
import pandas as pd

from vae_ld.learning_dynamics import logger


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
        self._debiased = debiased

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
        if self._debiased:
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

    def cka(self, kc, lc):
        # Note: this method assumes that kc and lc are the centered kernel values given by cka.center(cka.kernel(.))
        # Compute tr(KcLc) = vec(kc)^T vec(lc), omitting the term (m-1)**2, which is canceled by CKA
        hsic = np.dot(kc.ravel(), lc.ravel())
        # Divide by the product of the Frobenius norms of kc and lc to get CKA
        return hsic / (np.linalg.norm(kc, ord="fro") * np.linalg.norm(lc, ord="fro"))

    def __call__(self, x, y):
        return self.cka(x, y)


def get_activations(data, model_path):
    model = tf.keras.models.load_model(model_path)
    acts = model.encoder.predict(data)
    acts += model.decoder.predict(acts[-1])
    # Note that one could get weights using l.get_weights() instead of l.name here
    layer_names = [l.name for l in model.encoder.layers]
    layer_names += [l.name for l in model.decoder.layers]
    return model, acts, layer_names


def prepare_activations(cka, x):
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    kc = cka.center(cka.kernel(x))
    return kc


def compute_models_cka(cka, data, m1_path, m2_path, save_path, models_info):
    m1, acts1, layers1 = get_activations(data, m1_path)
    m2, acts2, layers2 = get_activations(data, m2_path)
    res = {}
    for i, l1 in enumerate(layers1):
        logger.info("Preparing layer {} of {}".format(l1, m1_path))
        kc = prepare_activations(cka, acts1[i])
        res[l1] = {}
        for j, l2 in enumerate(layers2):
            logger.info("Preparing layer {} of {}".format(l2, m2_path))
            lc = prepare_activations(cka, acts2[j])
            logger.info("Computing CKA({}, {})".format(l1, l2))
            res[l1][l2] = cka(kc, lc)
    res = pd.DataFrame(res).T
    for k, v in models_info.items():
        res[k] = v
    # Save csv with m1 layers as header, m2 layers as indexes
    res = res.rename_axis("m1", axis="columns")
    res = res.rename_axis("m2")
    res.to_csv(save_path, sep="\t")

