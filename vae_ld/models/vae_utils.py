from math import pi
import tensorflow as tf
from tensorflow import math as tfm


def compute_gaussian_log_pdf(z, z_mean, z_log_var):
    """ Compute the log probability density of a Gaussian distribution. Based on Locatello et al.
    `implementation <https://github.com/google-research/disentanglement_lib>`_

    Parameters
    ----------
    z : tf.Tensor
        The sampled values
    z_mean : tf.Tensor
        The mean of the Gaussian
    z_log_var : tf.Tensor
        The log variance of the Gaussian

    Returns
    -------
    tf.Tensor
        The log probability density

    Examples
    --------
    >>> mean = tf.constant([[0., 0.],[0., 0.]])
    >>> log_var = tf.constant([[0., 0.],[0., 0.]])
    >>> z = tf.constant([[0., 0.],[0., 0.]])
    >>> compute_gaussian_log_pdf(z, log_var, mean)
        <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[-0.9189385, -0.9189385],
               [-0.9189385, -0.9189385]], dtype=float32)>
    >>> mean, log_var, z = tf.constant([[2., -1.]]), tf.constant([[0., 0.]]), tf.constant([[-1., 2.]])
    >>> compute_gaussian_log_pdf(z, log_var, mean)
        <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[-3.0743961, -3.0743961]], dtype=float32)>
    """
    log2pi = tfm.log(2. * tf.constant(pi))
    return -0.5 * (tfm.square(z - z_mean) * tfm.exp(-z_log_var) + z_log_var + log2pi)


def compute_covariance(x):
    r""" Compute the covariance matrix of `x` based on Locatello et al.
    `implementation <https://github.com/google-research/disentanglement_lib>`_

    Uses

    .. math::
       cov(X) = E[XX^T] - E[X]E[X]^T.

    Parameters
    ----------
    x : tf.Tensor
        A (n_examples, n_features) matrix

    Returns
    -------
    tf.Tensor
        The (n_features, n_features) matrix
    """
    e_x = tfm.reduce_mean(x, axis=0)
    e_x_e_xt = tf.expand_dims(e_x, 1) * tf.expand_dims(e_x, 0)
    e_xxt = tfm.reduce_mean(tf.expand_dims(x, 2) * tf.expand_dims(x, 1), axis=0)
    return tfm.subtract(e_xxt, e_x_e_xt)


def compute_batch_tc(z, z_mean, z_log_var):
    r""" Estimates the total correlation over a batch. Based on Locatello et al.
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    Compute

    .. math::
       E_j [\log q(z(x_j)) - \log \prod_l q(z(x_j)_l)]

    where i and j are indexing the batch size and l is indexing
    the number of latent factors.

    Parameters
    ----------
    z : tf.Tensor
        The sampled values
    z_mean : tf.Tensor
        The mean of the Gaussian
    z_log_var : tf.Tensor
        The log variance of the Gaussian

    Returns
    -------
    tf.Tensor
        The total correlation estimated over the batch

    Examples
    --------
    >>> mean = tf.constant([[0., 0.],[0., 0.]])
    >>> log_var = tf.constant([[0., 0.],[0., 0.]])
    >>> z = tf.constant([[0., 0.],[0., 0.]])
    >>> compute_batch_tc(z, log_var, mean)
        <tf.Tensor: shape=(), dtype=float32, numpy=-0.6931472>
    >>> mean, log_var, z = tf.constant([[2., -1.]]), tf.constant([[0., 0.]]), tf.constant([[-1., 2.]])
    >>> compute_batch_tc(z, mean, log_var)
        <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
    """
    log_qz = compute_gaussian_log_pdf(tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0), tf.expand_dims(z_log_var, 0))
    prod_log_qz = tfm.reduce_sum(tfm.reduce_logsumexp(log_qz, axis=1, keepdims=False), axis=1, keepdims=False)
    log_sum_qz = tfm.reduce_logsumexp(tfm.reduce_sum(log_qz, axis=2, keepdims=False), axis=1, keepdims=False)
    return tfm.reduce_mean(log_sum_qz - prod_log_qz)


def shuffle_z(z):
    """ Shuffle the latent variables of a batch.
    Based on Locatello et al. `implementation <https://github.com/google-research/disentanglement_lib>`_.

    Parameters
    ----------
    z : tf.Tensor
        The sampled values

    Returns
    -------
    tf.Tensor
        The shuffled sampled values

    Examples
    --------
    >>> z = tf.constant([[1., 2.],[3., 4.]])
    >>> shuffle_z(z)
        <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[3., 2.],
               [1., 4.]], dtype=float32)>
    """
    # Use a frozen variable to prevent tracking of shuffled values from gradient tape
    z_frozen = tf.Variable(z, trainable=False)
    shuffled = [tf.random.shuffle(z_frozen[:, i]) for i in range(tf.shape(z_frozen)[1])]
    return tf.stack(shuffled, 1, name="z_shuffled")
