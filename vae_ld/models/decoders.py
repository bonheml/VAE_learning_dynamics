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


class DeepConvDecoder(tf.keras.Model):
    """ Deeper convolutional decoder. Each Convolutional block is composed of n convolutional layers where the last
    have a stride of 2 and the other have a stride of 1 (and thus the same output shape as the previous layers in the
    block). The fully connected block is composed of n fully connected layers where the output size is multiplied by 2
    after each iteration.
    """

    def __init__(self, input_shape, output_shape):
        super(DeepConvDecoder, self).__init__()
        # Reverse of FC Block
        self.block_1 = self._build_fc_block(4, 512, "decoder/1", input_shape=input_shape)

        # Reshape to 3D
        self.reshape = layers.Reshape((2, 2, 512), name="decoder/reshape")

        # Reverse of Conv Blocks 4 to 1
        self.block_2 = self._build_conv_block(4, 256, "decoder/2")
        self.block_3 = self._build_conv_block(4, 128, "decoder/3")
        self.block_4 = self._build_conv_block(2, 64, "decoder/4")
        self.block_5 = self._build_conv_block(2, 32, "decoder/5")

        # Reducing the number of filter to the original image channel number
        self.d6 = layers.Conv2DTranspose(filters=output_shape[2], kernel_size=4, strides=1, padding="same",
                                         name="decoder/output")

    def _build_conv_block(self, n, filters, name, kernel_size=4, activation="relu", padding="same"):
        block = []
        strides = 1
        for i in range(n):
            if (i + 1) == n:
                strides += 1
            block.append(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding,
                                       strides=strides, name="{}{}".format(name, i + 1)))
        return block

    def _build_fc_block(self, n, start_size, name, activation="relu", input_shape=None):
        block = []
        for i in range(n):
            if i == 0 and input_shape is not None:
                block.append(layers.Dense(start_size, activation=activation, name="{}{}".format(name, i + 1),
                                          input_shape=input_shape))
            else:
                block.append(layers.Dense(start_size, activation=activation, name="{}{}".format(name, i + 1)))
            start_size *= 2
        return block

    def _iterate_on_block(self, inputs, block):
        x = inputs
        for i in range(len(block)):
            x = block(x)
        return x

    def call(self, inputs):
        # Reverse of FC Block
        x1 = self._iterate_on_block(inputs, self.block_1)

        # Reshape to 3D
        x1r = self.reshape(x1)

        # Reverse of Conv blocks 4 - 1
        x2 = self._iterate_on_block(x1r, self.block_2)
        x3 = self._iterate_on_block(x2, self.block_3)
        x4 = self._iterate_on_block(x3, self.block_4)
        x5 = self._iterate_on_block(x4, self.block_5)

        # Reducing the number of filter to the original image channel number
        out = self.d6(x5)

        # We only return the activation at the end of each block + FC layers
        return x1, x2, x3, x4, x5, out


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
