import tensorflow as tf
from tensorflow.keras import layers


def fc_discriminator(input_shape, show_all):
    """ Fully connected discriminator initially used in FactorVAE. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

    :param input_shape: shape of the latent representations
    :param show_all: if True, track all the layer outputs else, only the mean, variance and sampled latent.
    :return: the decoder
    """
    inputs = tf.keras.Input(shape=input_shape)
    adv1 = layers.Flatten(name="adv1")(inputs)
    adv2 = layers.LeakyReLU(1000, name="adv2")(adv1)
    adv3 = layers.LeakyReLU(1000, name="adv3")(adv2)
    adv4 = layers.LeakyReLU(1000, name="adv4")(adv3)
    adv5 = layers.LeakyReLU(1000, name="adv5")(adv4)
    adv6 = layers.LeakyReLU(1000, name="adv6")(adv5)
    logits = layers.Dense(2, activation=None, name="logits")(adv6)
    output = layers.Softmax()(logits)
    if show_all is True:
        return tf.keras.Model(inputs=inputs, outputs=[adv1, adv2, adv3, adv4, adv5, adv6, logits, output],
                              name="discriminator")
    return tf.keras.Model(inputs=inputs, outputs=[output], name="discriminator")
