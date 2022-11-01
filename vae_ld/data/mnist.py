from collections import Iterable

import numpy as np
from vae_ld.data.dataset import Data
from vae_ld.data.util import load_and_preprocess_tf_dataset


class MnistIndex:
    def __init__(self, labels):
        self._index = np.array(labels)

    def index_to_features(self, idxs):
        return self._index[idxs]


class Mnist(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data, self._features, self._lookup_table = self.load_data()
        self.index = MnistIndex(self._features)

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
        mnist_digits, mnist_features = load_and_preprocess_tf_dataset("mnist", self.path, [64, 64])
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


class MnistSvhn(Mnist):

    def __getitem__(self, keys):
        if not isinstance(keys, Iterable):
            keys = [keys]
        return [self.data[k] for k in keys]

    def load_data(self):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
        mnist_digits, mnist_features = load_and_preprocess_tf_dataset("mnist", self.path, [64, 64])
        svhn_digits, svhn_features = load_and_preprocess_tf_dataset("svhn_cropped", self.path, [64, 64])
        lookup_table = []
        data = []
        labels = []
        for i in range(10):
            mnist_idx = np.where(mnist_features == i)[0]
            svhn_idx = np.where(svhn_features == i)[0]
            joint = list(zip(mnist_idx, svhn_idx))
            lookup_table.append(joint)
            data += [(mnist_digits[j], svhn_digits[k]) for j, k in joint]
            labels += [i for _ in range(len(joint))]
        return data, labels, lookup_table
