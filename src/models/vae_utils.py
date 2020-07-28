from math import pi
import tensorflow as tf
from tensorflow import math as tfm


def compute_gaussian_kl(z_log_var, z_mean):
    """ Compute the KL divergence between a Gaussian and a Normal distribution. Based on Locatello et al.
    implementation (https://github.com/google-research/disentanglement_lib)

    :param z_log_var: the log variance of the Gaussian
    :param z_mean: the mean of the Gaussian
    :return: the KL divergence
    """
    kl_loss = tfm.square(z_mean) + tfm.exp(z_log_var) - z_log_var - 1
    return tfm.reduce_mean(0.5 * tfm.reduce_sum(kl_loss, [1]), name="kl_loss")


def compute_gaussian_log_pdf(z, z_mean, z_log_var):
    """ Compute the log probability density of a Gaussian distribution. Based on Locatello et al. implementation
    (https://github.com/google-research/disentanglement_lib)

    :param z: the sampled values
    :param z_mean: the mean of the Gaussian
    :param z_log_var: the log variance of the Gaussian
    :return: the log probability density
    """
    log2pi = tfm.log(2. * tf.constant(pi))
    return -0.5 * (tfm.square(z - z_mean) * tfm.exp(-z_log_var) + z_log_var + log2pi)


def compute_covariance(x):
    """ Compute the covariance cov(x) = E[x*x^T] - E[x]E[x]^T. Based on Locatello et al. implementation
    (https://github.com/google-research/disentanglement_lib)

    :param x: a matrix of size N*M
    :return: the covariance of x, a matrix of size M*M
    """
    e_x = tfm.reduce_mean(x, axis=0)
    e_x_e_xt = tf.expand_dims(e_x, axis=1) * tf.expand_dims(e_x, axis=0)
    e_xxt = tfm.reduce_mean(tf.expand_dims(x, 2), tf.expand_dims(x, 1), axis=0)
    return tfm.subtract(e_xxt, e_x_e_xt)


def compute_batch_tc(z, z_mean, z_log_var):
    """ Estimates the total correlation over a batch. Based on Locatello et al. implementation
    (https://github.com/google-research/disentanglement_lib).
    Compute E_j[log(q(z(x_j))) - log(prod_l q(z(x_j)_l))] where i and j are indexing the batch size and l is indexing
    the number of latent factors.

    :param z: the sampled values
    :param z_mean: the mean of the Gaussian
    :param z_log_var: the log variance of the Gaussian
    :return: the total correlation estimated over the batch
    """
    log_qz = compute_gaussian_log_pdf(tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0), tf.expand_dims(z_log_var, 0))
    prod_log_qz = tfm.reduce_sum(tfm.reduce_logsumexp(log_qz, axis=1, keepdims=False), axis=1, keepdims=False)
    log_sum_qz = tfm.reduce_logsumexp(tfm.reduce_sum(log_qz, axis=2, keepdims=False), axis=1, keepdims=False)
    return tfm.reduce_mean(log_sum_qz, prod_log_qz)


def shuffle_z(z):
    """ Shuffle the latent variables of a batch. The values of the latent varuBased on Locatello et al. implementation
    (https://github.com/google-research/disentanglement_lib).

    :param z: the latent representations of size batch_size * num_latent
    :return: the shuffled representations of size batch_size * num_latent
    """
    shuffled = [tf.random.shuffle(z[:, i]) for i in range(tf.shape(z)[1])]
    return tf.stack(shuffled, 1, name="z_shuffled")
