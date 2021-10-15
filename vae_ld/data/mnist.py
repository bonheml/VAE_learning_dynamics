import numpy as np
import tensorflow as tf

class Mnist:
    def __init__(self, *, observation_shape, **kwargs):
        self._data = None
        self._observation_shape = tuple(observation_shape)

    @property
    def data(self):
        return self._data

    def load_data(self):
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
        self._data = mnist_digits
        return mnist_digits
