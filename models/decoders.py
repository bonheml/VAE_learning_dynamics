import tensorflow as tf
from pandas import np
from tensorflow.keras import layers


class DeconvolutionalDecoder:

    def __init__(self, *, input_shape, output_shape, save_activations):
        """ Deconvolutional decoder initially used in beta-VAE [1]. Based on Locatello et al. [2] implementation
        (https://github.com/google-research/disentanglement_lib)

        [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
        In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
        [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
        Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
        on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
        pp. 4114–4124.

        :param input_shape: shape of the latent representations
        :param output_shape: shape of the output
        :param save_activations: if True, track all the layer outputs else, only the mean, variance and sampled latent.
        :return: the decoder
        """
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.save_activations = save_activations

    def build(self):

        inputs = tf.keras.Input(shape=self.input_shape)
        d1 = layers.Dense(256, activation="relu", name="d1")(inputs)
        d2 = layers.Dense(1024, activation="relu", name="d2")(d1)
        d3 = layers.Reshape((-1, 4, 4, 64))(d2)
        d4 = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, activation="relu", padding="same",
                                    name="d3")(d3)
        d5 = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="same",
                                    name="d4")(d4)
        d6 = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="same",
                                    name="d5")(d5)
        d7 = layers.Conv2DTranspose(filters=self.output_shape[2], kernel_size=4, strides=2, activation="relu",
                                    padding="same", name="d6")(d6)
        output = layers.Reshape((-1,) + self.output_shape, name="output")(d7)
        if self.save_activations is True:
            return tf.keras.Model(inputs=inputs, outputs=[d1, d2, d3, d4, d5, d6, d7, output], name="decoder")
        return tf.keras.Model(inputs=inputs, outputs=[output], name="decoder")


class FullyConnectedDecoder:

    def __init__(self, *, input_shape, output_shape, save_activations):
        """ Fully connected decoder initially used in beta-VAE [1]. Based on Locatello et al. [2] implementation
        (https://github.com/google-research/disentanglement_lib)

        [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
        In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
        [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
        Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
        on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
        pp. 4114–4124.

        :param input_shape: shape of the latent representations
        :param output_shape: shape of the output
        :param save_activations: if True, track all the layer outputs else, only the mean, variance and sampled latent.
        :return: the decoder
        """
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.save_activations = save_activations

    def build(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        d1 = layers.Dense(1200, activation="tanh", name="d1")(inputs)
        d2 = layers.Dense(1200, activation="tanh", name="d2")(d1)
        d3 = layers.Dense(1200, activation="tanh", name="d3")(d2)
        d4 = layers.Dense(np.prod(self.output_shape), activation=None, name="d4")(d3)
        output = layers.Reshape(self.output_shape, name="output")(d4)
        if self.save_activations is True:
            return tf.keras.Model(inputs=inputs, outputs=[d1, d2, d3, d4, output], name="decoder")
        return tf.keras.Model(inputs=inputs, outputs=[output], name="decoder")
