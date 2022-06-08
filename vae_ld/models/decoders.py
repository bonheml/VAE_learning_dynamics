import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class DeconvolutionalDecoder(tf.keras.Model):
    """ Deconvolutional decoder initially used in beta-VAE [1]. Based on Locatello et al. [2]
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    References
    ----------
    .. [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017).
           β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In 5th International
           Conference on Learning Representations, ICLR 2017, Toulon, France.
    .. [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
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


class VGG19Decoder(tf.keras.Model):
    """ Convolutional decoder based on VGG19 architecture. Based on the encoder version proposed by [1] and
    adapted for VGG19 with some an additional fully connected block.

    References
    ----------
    .. [1] Kebir, A., Taibi, M., & Serradilla, F. (2021). Compressed VGG16 Auto-Encoder for Road Segmentation
     from Aerial Images with Few Data Training.
    """

    def __init__(self, input_shape, output_shape):
        super(VGG19Decoder, self).__init__(name="vgg_19_decoder")
        # Reverse of FC Block
        # The first FC layer has a lower dimensionality than in the encoder
        # because the next FC layer is only 2048
        self.d11 = layers.Dense(1024, activation='relu', name='decoder/11', input_shape=(input_shape,))
        # This is equivalent to Dense + Flatten in the decoder
        self.d12 = layers.Dense(2048, activation='relu', name='decoder/22')
        self.d13 = layers.Reshape((2, 2, 512), name="decoder/reshape")

        # Reverse of Block 4
        # Use default (2,2) up sampling size
        self.d21 = layers.UpSampling2D(name='decoder/21')
        self.d22 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', name='decoder/22')
        self.d23 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', name='decoder/23')
        self.d24 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', name='decoder/24')
        self.d25 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', name='decoder/25')

        # Reverse of Block 3
        self.d31 = layers.UpSampling2D(name='decoder/31')
        self.d32 = layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', name='decoder/32')
        self.d33 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', name='decoder/33')
        self.d34 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', name='decoder/34')
        self.d35 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', name='decoder/35')

        # Reverse of Block 2
        self.d41 = layers.UpSampling2D(name='decoder/41')
        self.d42 = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='decoder/42')
        self.d43 = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='decoder/43')

        # Reverse of Block 1
        self.d51 = layers.UpSampling2D(name='decoder/51')
        self.d52 = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='decoder/52')
        self.d53 = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='decoder/53')

        # Resizing to the correct output size
        self.d61 = layers.Flatten(name='decoder/51')
        self.d62 = layers.Dense(np.prod(output_shape), activation='relu', name='decoder/52')
        self.d63 = layers.Reshape(output_shape, name='decoder/53')

    def call(self, inputs):
        # Reverse of FC Block
        x11 = self.d11(inputs)
        x12 = self.d12(x11)
        x13 = self.d13(x12)

        # Reverse of Block 4
        x21 = self.d21(x13)
        x22 = self.d22(x21)
        x23 = self.d23(x22)
        x24 = self.d24(x23)
        x25 = self.d25(x24)

        # Reverse of Block 3
        x31 = self.d31(x25)
        x32 = self.d32(x31)
        x33 = self.d33(x32)
        x34 = self.d34(x33)
        x35 = self.d35(x34)

        # Reverse of Block 2
        x41 = self.d41(x35)
        x42 = self.d42(x41)
        x43 = self.d43(x42)

        # Reverse of Block 1
        x51 = self.d51(x43)
        x52 = self.d52(x51)
        x53 = self.d53(x52)

        # Resizing to the correct output size
        x61 = self.d61(x53)
        x62 = self.d62(x61)
        x63 = self.d63(x62)

        # We only return the activation at the end of each block + FC layers
        return x11, x12, x25, x35, x43, x53, x62, x63


class FullyConnectedDecoder(tf.keras.Model):
    """ Fully connected decoder initially used in beta-VAE [1]. Based on Locatello et al. [2]
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    References
    ----------
    .. [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017).
           β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In 5th International
           Conference on Learning Representations, ICLR 2017, Toulon, France.
    .. [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
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
