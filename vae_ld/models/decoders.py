import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class DeconvolutionalDecoder(tf.keras.Model):
    """ Deconvolutional decoder initially used in beta-VAE [1]. Based on Locatello et al. [2] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
    """
    def __init__(self, input_shape, output_shape):
        super(DeconvolutionalDecoder, self).__init__()
        self.d1 = layers.Dense(256, activation="relu", name="decoder/1", input_shape=(input_shape,))
        self.d2 = layers.Dense(1024, activation="relu", name="decoder/2")
        self.d3 = layers.Reshape((4, 4, 64), name="decoder/reshape")
        self.d4 = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, activation="relu", padding="same",
                                         name="decoder/3")
        self.d5 = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="same",
                                         name="decoder/4")
        self.d6 = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="same",
                                         name="decoder/5")
        self.d7 = layers.Conv2DTranspose(filters=output_shape[2], kernel_size=4, strides=2, padding="same",
                                         name="decoder/6")
        self.d8 = layers.Reshape(output_shape, name="decoder/output")

    def call(self, inputs):
        x1 = self.d1(inputs)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        x6 = self.d6(x5)
        x7 = self.d7(x6)
        x8 = self.d8(x7)
        return x1, x2, x3, x4, x5, x6, x7, x8


class FullyConnectedDecoder(tf.keras.Model):
    """ Fully connected decoder initially used in beta-VAE [1]. Based on Locatello et al. [2] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
    """
    def __init__(self, input_shape, output_shape):
        super(FullyConnectedDecoder, self).__init__()
        self.d1 = layers.Dense(1200, activation="tanh", name="decoder/1", input_shape=(input_shape,))
        self.d2 = layers.Dense(1200, activation="tanh", name="decoder/2")
        self.d3 = layers.Dense(1200, activation="tanh", name="decoder/3")
        self.d4 = layers.Dense(np.prod(output_shape), activation=None, name="decoder/4")
        self.d5 = layers.Reshape(output_shape, name="decoder/output")

    def call(self, inputs):
        x1 = self.d1(inputs)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        return x1, x2, x3, x4, x5


class MnistDecoder(tf.keras.Model):
    """ Deconvolutional decoder initially used in Keras VAE tutorial for mnist data.
    (https://keras.io/examples/generative/vae/#define-the-vae-as-a-model-with-a-custom-trainstep)
    """
    def __init__(self, input_shape, output_shape):
        super(MnistDecoder, self).__init__()
        self.d1 = layers.Dense(7 * 7 * 64, activation="relu", name="decoder/1", input_shape=(input_shape,))
        self.d2 = layers.Reshape((7, 7, 64), name="decoder/2")
        self.d3 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name="decoder/3")
        self.d4 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name="decoder/4")
        self.d5 = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name="decoder/5")
        self.d6 = layers.Reshape(output_shape, name="decoder/output")

    def call(self, inputs):
        x1 = self.d1(inputs)
        x2 = self.d2(x1)
        x3 = self.d2(x2)
        x4 = self.d2(x3)
        x5 = self.d2(x4)
        return x1, x2, x3, x4, x5
