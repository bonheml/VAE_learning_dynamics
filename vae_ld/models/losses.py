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
