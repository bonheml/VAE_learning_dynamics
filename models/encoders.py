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


def conv_encoder(input_shape, output_shape, show_all):
    """ Convolutional encoder initially used in beta-VAE. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

    :param input_shape: shape of the input
    :param output_shape: shape of the latent representations
    :param show_all: if True, track all the layer outputs else, only the mean, variance and sampled latent.
    :return: the encoder
    """
    inputs = tf.keras.Input(shape=input_shape)
    e1 = layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu", padding="same", name="e1")(inputs)
    e2 = layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu", padding="same", name="e2")(e1)
    e3 = layers.Conv2D(filters=64, kernel_size=2, strides=2, activation="relu", padding="same", name="e3")(e2)
    e4 = layers.Conv2D(filters=64, kernel_size=2, strides=2, activation="relu", padding="same", name="e4")(e3)
    e5 = layers.Flatten(name="e5")(e4)
    e6 = layers.Dense(256, activation="relu", name="e6")(e5)
    z_mean = layers.Dense(output_shape, name="z_mean")(e6)
    z_log_var = layers.Dense(output_shape, name="z_log_var")(e6)
    z = Sampling()([z_mean, z_log_var])
    if show_all is True:
        return tf.keras.Model(inputs=inputs, outputs=[e1, e2, e3, e4, e5, e6, z_mean, z_log_var, z], name="encoder")
    return tf.keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")


def fc_encoder(input_shape, output_shape, show_all):
    """ Fully connected encoder initially used in beta-VAE. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

    :param input_shape: shape of the input
    :param output_shape: shape of the latent representations
    :param show_all: if True, track all the layer outputs else, only the mean, variance and sampled latent.
    :return: the encoder
    """
    inputs = tf.keras.Input(shape=input_shape)
    e1 = layers.Flatten(name="e1")(input_shape)
    e2 = layers.Dense(1200, activation="relu", name="e2")(e1)
    e3 = layers.Dense(1200, activation="relu", name="e3")(e2)
    z_mean = layers.Dense(output_shape, activation=None, name="z_mean")(e1)
    z_log_var = layers.Dense(output_shape, activation=None, name="z_log_var")(e1)
    z = Sampling()([z_mean, z_log_var])
    if show_all is True:
        return tf.keras.Model(inputs=inputs, outputs=[e1, e2, e3, z_mean, z_log_var, z], name="encoder")
    return tf.keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")

