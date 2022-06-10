import tensorflow as tf
from tensorflow import math as tfm


class KLD:
    """ Compute the KL divergence between a multivariate Gaussian and a standard multivariate Gaussian distribution.
    Based on Locatello et al.`implementation <https://github.com/google-research/disentanglement_lib>`_

    Parameters
    ----------
    z_log_var : tf.Tensor
        The log variance of the Gaussian
    z_mean : tf.Tensor
        The mean of the Gaussian

    Returns
    -------
    tf.Tensor
        The KL divergence

    Examples
    --------
    >>> kld = KLD()
    >>> mean = tf.constant([[0., 0.],[0., 0.]])
    >>> log_var = tf.constant([[0., 0.],[0., 0.]])
    >>> kld(log_var, mean)
        <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 0.], dtype=float32)>
    >>> mean, log_var = tf.constant([[2., -1.]]), tf.constant([[0., 0.]])
    >>> kld(log_var, mean)
        <tf.Tensor: shape=(1,), dtype=float32, numpy=array([2.5], dtype=float32)>
    """

    def __call__(self, z_log_var, z_mean):
        kl_loss = tfm.square(z_mean) + tfm.exp(z_log_var) - z_log_var - 1
        return 0.5 * tfm.reduce_sum(kl_loss, [1])


class Hellinger:
    """ Compute the Hellinger distance between a multivariate Gaussian and a standard multivariate Gaussian distribution.
    Hellinger distance is a metric and

    .. math::
      0 \leq D_H(P,Q) \leq 1

    To avoid underflow, we multiply the results by 100 by default and get a score between 0 and 100

    Parameters
    ----------
    z_log_var : tf.Tensor
       The log variance of the Gaussian
    z_mean : tf.Tensor
       The mean of the Gaussian

    Returns
    -------
    tf.Tensor
       The Hellinger distance

    Examples
    --------
    >>> hd = Hellinger()
    >>> mean = tf.constant([[0., 0.],[0., 0.]])
    >>> log_var = tf.constant([[0., 0.],[0., 0.]])
    >>> hd(log_var, mean)
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 0.], dtype=float32)>
    >>> mean, log_var = tf.constant([[2., -1.]]), tf.constant([[0., 0.]])
    >>> hd(log_var, mean)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.68171734], dtype=float32)>
    """
    def __init__(self, mult=100):
        self.mult = mult

    def __call__(self, z_log_var, z_mean):
        z_var_plus_1 = tfm.exp(z_log_var) + 1
        n = z_mean.shape[1] / 2
        exp_term = tfm.reduce_sum(z_log_var - 2 * tfm.log(z_var_plus_1) - tfm.square(z_mean) / z_var_plus_1, axis=1)
        hellinger = tfm.sqrt(1 - 2 ** n * tfm.exp(0.25 * exp_term))
        # In case of underflow, replace with 0
        hellinger = tf.where(tfm.is_nan(hellinger), 0., hellinger)
        return self.mult * hellinger
