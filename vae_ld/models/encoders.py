import tensorflow as tf
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """ Uses the reparametrisation trick to sample z = mean + exp(0.5 * log_var) * eps where eps ~ N(0,I)
    """

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class GenericEncoder:
    def __init__(self, *, input_shape, output_shape, save_activations):
        """ Generic encoder initialisation
        :param input_shape: shape of the input
        :param output_shape: shape of the latent representations
        :param save_activations: if True, track all the layer outputs else, only the mean, variance and sampled latent
        """
        self.input_shape = tuple(input_shape)
        self.output_shape = output_shape
        self.save_activations = save_activations

    def build(self):
        raise NotImplementedError()


class ConvolutionalEncoder(GenericEncoder):
    """ Convolutional encoder initially used in beta-VAE [1]. Based on Locatello et al. [2] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
    """

    def build(self):
        inputs = tf.keras.Input(shape=self.input_shape, name="encoder/input")
        e1 = layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu", padding="same",
                           name="encoder/1")(inputs)
        e2 = layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu", padding="same",
                           name="encoder/2")(e1)
        e3 = layers.Conv2D(filters=64, kernel_size=2, strides=2, activation="relu", padding="same",
                           name="encoder/3")(e2)
        e4 = layers.Conv2D(filters=64, kernel_size=2, strides=2, activation="relu", padding="same",
                           name="encoder/4")(e3)
        e5 = layers.Flatten(name="encoder/5")(e4)
        e6 = layers.Dense(256, activation="relu", name="encoder/6")(e5)
        z_mean = layers.Dense(self.output_shape, name="encoder/z_mean")(e6)
        z_log_var = layers.Dense(self.output_shape, name="encoder/z_log_var")(e6)
        z = Sampling(name="encoder/z")([z_mean, z_log_var])
        if self.save_activations is True:
            return tf.keras.Model(inputs=inputs, outputs=[e1, e2, e3, e4, e5, e6, z_mean, z_log_var, z], name="encoder")
        return tf.keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")


class FullyConnectedEncoder(GenericEncoder):
    """ Fully connected encoder initially used in beta-VAE [1]. Based on Locatello et al. [2] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
    """

    def build(self):
        inputs = tf.keras.Input(shape=self.input_shape, name="encoder/input")
        e1 = layers.Flatten(name="encoder/1")(inputs)
        e2 = layers.Dense(1200, activation="relu", name="encoder/2")(e1)
        e3 = layers.Dense(1200, activation="relu", name="encoder/3")(e2)
        z_mean = layers.Dense(self.output_shape, activation=None, name="encoder/z_mean")(e3)
        z_log_var = layers.Dense(self.output_shape, activation=None, name="encoder/z_log_var")(e3)
        z = Sampling(name="encoder/z")([z_mean, z_log_var])
        if self.save_activations is True:
            return tf.keras.Model(inputs=inputs, outputs=[e1, e2, e3, z_mean, z_log_var, z], name="encoder")
        return tf.keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")


class MnistEncoder(GenericEncoder):
    """ Convolutional encoder initially used in Keras VAE tutorial for mnist data.
    (https://keras.io/examples/generative/vae/#define-the-vae-as-a-model-with-a-custom-trainstep)
    """

    def build(self):
        inputs = tf.keras.Input(shape=self.input_shape, name="encoder/input")
        e1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same", name="encoder/1")(inputs)
        e2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", name="encoder/2")(e1)
        e3 = layers.Flatten(name="encoder/3")(e2)
        e4 = layers.Dense(16, activation="relu", name="encoder/4")(e3)
        z_mean = layers.Dense(self.output_shape, activation=None, name="encoder/z_mean")(e4)
        z_log_var = layers.Dense(self.output_shape, activation=None, name="encoder/z_log_var")(e4)
        z = Sampling(name="encoder/z")([z_mean, z_log_var])
        if self.save_activations is True:
            return tf.keras.Model(inputs=inputs, outputs=[e1, e2, e3, e4, z_mean, z_log_var, z], name="encoder")
        return tf.keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")
