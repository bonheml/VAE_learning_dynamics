import numpy as np
import tensorflow as tf
from vae_ld.data.dataset import Data


class Mnist(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data, self._features, self._lookup_table = self.load_data()

    def load_data(self):
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
        mnist_digits = np.concatenate([X_train, X_test], axis=0)
        mnist_features = np.concatenate([Y_train, Y_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255.
        lookup_table = []
        for i in range(10):
            lookup_table.append(np.where(mnist_features == i)[0])
        return mnist_digits, mnist_features, lookup_table

    def sample_factors(self, batch_size, seed):
        return seed.choice(10, batch_size)

    def sample_observations_from_factors(self, factors, seed):
        indices = []
        for i in factors:
            indices.append(self._lookup_table[i][seed.choice(len(self._lookup_table[i]), 1)][0])
        return self._data[indices]
