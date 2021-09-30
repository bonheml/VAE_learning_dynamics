import tensorflow as tf
from tensorflow.keras import layers


class FullyConnectedDiscriminator(tf.keras.Model):
    """" Fully connected discriminator initially used in FactorVAE. Based on Locatello et al. [1] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

    :param input_shape: shape of the latent representations
    :return: the discriminator
    """

    def __init__(self, input_shape):
        super(FullyConnectedDiscriminator, self).__init__()
        self.adv1 = layers.Flatten(name="discriminator/1", input_shape=input_shape)
        self.adv2 = layers.Dense(1000, activation=layers.LeakyReLU(alpha=0.2), name="discriminator/2")
        self.adv3 = layers.Dense(1000, activation=layers.LeakyReLU(alpha=0.2), name="discriminator/3")
        self.adv4 = layers.Dense(1000, activation=layers.LeakyReLU(alpha=0.2), name="discriminator/4")
        self.adv5 = layers.Dense(1000, activation=layers.LeakyReLU(alpha=0.2), name="discriminator/5")
        self.adv6 = layers.Dense(1000, activation=layers.LeakyReLU(alpha=0.2), name="discriminator/6")
        self.adv7 = layers.Dense(1000, activation=layers.LeakyReLU(alpha=0.2), name="discriminator/7")
        self.logits = layers.Dense(2, activation=None, name="discriminator/logits")

    def call(self, inputs):
        x1 = self.adv1(inputs)
        x2 = self.adv2(x1)
        x3 = self.adv3(x2)
        x4 = self.adv4(x3)
        x5 = self.adv5(x4)
        x6 = self.adv6(x5)
        x7 = self.adv7(x6)
        logits = self.logits(x7)
        probs = tf.nn.softmax(logits)
        return x1, x2, x3, x4, x5, x6, x7, logits, probs
