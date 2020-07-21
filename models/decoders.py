import tensorflow as tf
from pandas import np
from tensorflow.keras import layers


def deconv_decoder(input_shape, output_shape, show_all):
    """ Deconvolutional decoder initially used in beta-VAE. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

    :param input_shape: shape of the latent representations
    :param output_shape: shape of the output
    :param show_all: if True, track all the layer outputs else, only the mean, variance and sampled latent.
    :return: the decoder
    """
    inputs = tf.keras.Input(shape=input_shape)
    d1 = layers.Dense(256, activation="relu", name="d1")(inputs)
    d2 = layers.Dense(1024, activation="relu", name="d2")(inputs)
    d3 = layers.Reshape((-1, 4, 4, 64))(d2)
    d4 = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, activation="relu", padding="same", name="d3")(d3)
    d5 = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="same", name="d4")(d4)
    d6 = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="same", name="d5")(d5)
    d7 = layers.Conv2DTranspose(filters=output_shape[2], kernel_size=4, strides=2, activation="relu",
                                padding="same", name="d6")(d6)
    output = layers.Reshape((-1,) + output_shape, name="output")(d7)
    if show_all is True:
        return tf.keras.Model(inputs=inputs, outputs=[d1, d2, d3, d4, d5, d6, d7, output], name="decoder")
    return tf.keras.Model(inputs=inputs, outputs=[output], name="decoder")


def fc_decoder(input_shape, output_shape, show_all):
    """ Fully connected decoder initially used in beta-VAE. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

    :param input_shape: shape of the latent representations
    :param output_shape: shape of the output
    :param show_all: if True, track all the layer outputs else, only the mean, variance and sampled latent.
    :return: the decoder
    """
    inputs = tf.keras.Input(shape=input_shape)
    d1 = layers.Dense(1200, activation="tanh", name="d1")(inputs)
    d2 = layers.Dense(1200, activation="tanh", name="d2")(d1)
    d3 = layers.Dense(1200, activation="tanh", name="d3")(d2)
    d4 = layers.Dense(np.prod(output_shape), activation=None, name="d4")(d3)
    output = layers.Reshape((-1,) + output_shape, name="output")(d4)
    if show_all is True:
        return tf.keras.Model(inputs=inputs, outputs=[d1, d2, d3, d4, output], name="decoder")
    return tf.keras.Model(inputs=inputs, outputs=[output], name="decoder")
