import tensorflow as tf
import numpy as np


class BernoulliLoss:
    def __call__(self, y_true, y_pred):
        flattened_dim = np.prod(y_true.get_shape().as_list()[1:])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        loss_flatten = tf.reshape(loss, shape=[-1, flattened_dim])
        return tf.reduce_sum(loss_flatten, axis=1)
