import tensorflow as tf
import numpy as np


class BernoulliLoss:
    """ Compute the Bernoulli Loss

    Parameters
    ----------
    y_true : tf.Tensor
        The matrix of (n_examples, n_features) true values
    y_pred : tf.Tensor
        The matrix of (n_examples, n_features) predicted values

    Returns
    -------
    tf.Tensor
        The array of n_examples bernoulli losses

    Examples
    --------
    >>> loss = BernoulliLoss()
    >>> y_true = tf.constant([[0., 0., 0., 1., 1., 1., 0.5]])
    >>> y_pred = tf.constant([[1., -1., 0., 1., -1., 0., 0.]])
    >>> loss(y_true, y_pred)
        <tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.3324885], dtype=float32)>
    """
    def __call__(self, y_true, y_pred):
        flattened_dim = np.prod(y_true.get_shape().as_list()[1:])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        loss_flatten = tf.reshape(loss, shape=[-1, flattened_dim])
        return tf.reduce_sum(loss_flatten, axis=1)


class SSIMLoss:
    """ Use the SSIM similarity metric [1] to compute a distance normalised in [0,1].
    We assume that pixel values are between 0 and 1. The distance computed is

    .. math::
       SSIM_loss(y_true, y_pred) = 1 - (SSIM(y_true, y_pred) + 1) / 2

    To avoid underflow, we multiply the results by 100 by default and get a score between 0 and 100

    Parameters
    ----------
    y_true : tf.Tensor
        The tensor of n_examples true images
    y_pred : tf.Tensor
        The tensor of n_examples predicted images

    Returns
    -------
    tf.Tensor
        The array of n_examples normalised ssim losses

    Examples
    --------
    >>> loss = SSIMLoss()
    >>> np.random.seed(0)
    >>> y_true = tf.constant(np.random.random((1,64,64,3)))
    >>> y_pred = tf.constant(np.random.random((1,64,64,3)))
    >>> loss(y_true, y_pred)
     <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.5001584], dtype=float32)>
    >>> loss(y_pred, y_true)
     <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.5001584], dtype=float32)>
    >>> loss(y_true, y_true)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>
    References
    ----------
    .. [1]  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment:
            from error visibility to structural similarity. IEEE transactions on image processing.
    """
    def __init__(self, mult=100):
        self.mult = mult

    def __call__(self, y_true, y_pred):
        norm_ssim = 1 - (tf.image.ssim(y_pred, y_true, 1.0) + 1) / 2
        return self.mult * norm_ssim
