import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from vae_ld.models import logger


class Sampling(layers.Layer):
    """ Sampling layer
    """

    def call(self, inputs, **kwargs):
        """ Uses the reparametrisation trick to sample z = mean + exp(0.5 * log_var) * eps
        where eps ~ N(0,I).

        Parameters
        ----------
        inputs : tuple
            Tuple of the form (z_mean, z_log_var)

        Returns
        -------
        tf.Tensor
            Sampled representation

        Examples
        --------
        >>> sampling = Sampling()
        >>> mean = tf.constant([[0., 0.],[0., 0.]])
        >>> log_var = tf.constant([[0., 0.],[0., 0.]])
        >>> sampling((mean, log_var))
        <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[-1.189146  ,  1.8920842 ],
               [-0.25569448,  0.48008046]], dtype=float32)>
        >>> sampling((mean, log_var))
        <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[ 0.78779936,  0.0514518 ],
               [-0.36348337, -0.65082115]], dtype=float32)>
        """
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class ConvolutionalEncoder(tf.keras.Model):
    """ Convolutional encoder initially used in beta-VAE [1]. Based on Locatello et al. [2]
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    References
    ----------
    .. [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017).
           β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In 5th International
           Conference on Learning Representations, ICLR 2017, Toulon, France.
    .. [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, input_shape, output_shape, zero_init=False):
        logger.debug("Expected input shape is {}".format(input_shape))
        super(ConvolutionalEncoder, self).__init__()
        self.e1 = layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=4, strides=2, activation="relu",
                                padding="same", name="encoder/1")
        self.e2 = layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu", padding="same",
                                name="encoder/2")
        self.e3 = layers.Conv2D(filters=64, kernel_size=2, strides=2, activation="relu", padding="same",
                                name="encoder/3")
        self.e4 = layers.Conv2D(filters=64, kernel_size=2, strides=2, activation="relu", padding="same",
                                name="encoder/4")
        self.e5 = layers.Flatten(name="encoder/5")
        self.e6 = layers.Dense(256, activation="relu", name="encoder/6")
        kernel_initializer = "zeros" if zero_init else "glorot_uniform"
        self.z_mean = layers.Dense(output_shape, name="encoder/z_mean", kernel_initializer=kernel_initializer)
        self.z_log_var = layers.Dense(output_shape, name="encoder/z_log_var", kernel_initializer=kernel_initializer)
        self.sampling = Sampling()

    def call(self, inputs):
        logger.debug("Received input shape is {}".format(inputs.shape))
        x1 = self.e1(inputs)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        z_mean = self.z_mean(x6)
        z_log_var = self.z_log_var(x6)
        z = self.sampling([z_mean, z_log_var])
        return x1, x2, x3, x4, x5, x6, z_mean, z_log_var, z


class DeepConvEncoder(tf.keras.Model):
    """ Deeper convolutional encoder. Each Convolutional block is composed of n convolutional layers where the first
    have a stride of 2 and the other have a stride of 1 (and thus the same output shape as the previous layers in the
    block). The fully connected block is composed of n fully connected layers where the output size is divided by 2
    after each iteration.
    """

    def __init__(self, input_shape, output_shape, zero_init=False):
        super(DeepConvEncoder, self).__init__()

        # Convolutional Blocks
        self.block_1 = self.build_conv_block(2, 32, "encoder/1", input_shape=input_shape)
        self.block_2 = self.build_conv_block(2, 64, "encoder/2")
        self.block_3 = self.build_conv_block(4, 128, "encoder/3")
        self.block_4 = self.build_conv_block(4, 256, "encoder/4")

        self.flatten = layers.Flatten(name='encoder/flatten')

        # Fully Connected Block
        self.block_5 = self.build_fc_block(4, 4096, "encoder/5")

        # Mean, variance, and sampling layers
        kernel_initializer = "zeros" if zero_init else "glorot_uniform"
        self.z_mean = layers.Dense(output_shape, name="encoder/z_mean", kernel_initializer=kernel_initializer)
        self.z_log_var = layers.Dense(output_shape, name="encoder/z_log_var", kernel_initializer=kernel_initializer)
        self.sampling = Sampling()

    def _iterate_on_block(self, inputs, block):
        x = inputs
        for i in range(len(block)):
            x = block(x)
        return x

    def _build_conv_block(self, n, filters, name, kernel_size=4, activation="relu", padding="same", input_shape=None):
        block = []
        strides = 2
        for i in range(n):
            if i == 0 and input_shape is not None:
                block.append(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation,
                                           padding=padding, strides=strides, name="{}{}".format(name, i + 1),
                                           input_shape=input_shape))
            else:
                block.append(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation,
                                           padding=padding, strides=strides, name="{}{}".format(name, i + 1)))
            if i == 0:
                strides -= 1
        return block

    def _build_fc_block(self, n, start_size, name, activation="relu"):
        block = []
        for i in range(n):
            block.append(layers.Dense(start_size, activation=activation, name="{}{}".format(name, i + 1)))
            start_size /= 2
        return block

    def call(self, inputs):
        x1 = self._iterate_on_block(inputs, self.block_1)
        x2 = self._iterate_on_block(x1, self.block_2)
        x3 = self._iterate_on_block(x2, self.block_3)
        x4 = self._iterate_on_block(x3, self.block_4)

        fx4 = self.flatten(x4)

        x5 = self._iterate_on_block(fx4, self.block_5)

        z_mean = self.z_mean(x5)
        z_log_var = self.z_log_var(x5)
        z = self.sampling([z_mean, z_log_var])

        # We only return the activation at the end of each block + FC layers and Sampling
        return x1, x2, x3, x4, x5, z_mean, z_log_var, z


class FullyConnectedEncoder(tf.keras.Model):
    """ Fully connected encoder initially used in beta-VAE [1]. Based on Locatello et al. [2]
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
        super(FullyConnectedEncoder, self).__init__()
        self.e1 = layers.Flatten(name="encoder/1", input_shape=input_shape)
        self.e2 = layers.Dense(1200, activation="relu", name="encoder/2")
        self.e3 = layers.Dense(1200, activation="relu", name="encoder/3")
        self.z_mean = layers.Dense(output_shape, activation=None, name="encoder/z_mean")
        self.z_log_var = layers.Dense(output_shape, activation=None, name="encoder/z_log_var")
        self.z = Sampling(name="encoder/z")

    def call(self, inputs):
        x1 = self.e1(inputs)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        z_mean = self.z_mean(x3)
        z_log_var = self.z_log_var(x3)
        z = self.z([z_mean, z_log_var])
        return x1, x2, x3, z_mean, z_log_var, z


class MnistEncoder(tf.keras.Model):
    """ Convolutional encoder initially used in Keras VAE tutorial for mnist data.
    (https://keras.io/examples/generative/vae/#define-the-vae-as-a-model-with-a-custom-trainstep)
    """

    def __init__(self, input_shape, output_shape):
        super(MnistEncoder, self).__init__()
        self.e1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same", name="encoder/1",
                                input_shape=input_shape)
        self.e2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", name="encoder/2")
        self.e3 = layers.Flatten(name="encoder/3")
        self.e4 = layers.Dense(16, activation="relu", name="encoder/4")
        self.z_mean = layers.Dense(output_shape, activation=None, name="encoder/z_mean")
        self.z_log_var = layers.Dense(output_shape, activation=None, name="encoder/z_log_var")
        self.z = Sampling(name="encoder/z")

    def call(self, inputs):
        x1 = self.e1(inputs)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x2)
        z_mean = self.z_mean(x4)
        z_log_var = self.z_log_var(x4)
        z = self.z([z_mean, z_log_var])
        return x1, x2, x3, x4, z_mean, z_log_var, z


class GONEncoder(tf.keras.Model):
    """ Encoder for GONs containing only mean, log variance and sampled representations
    """

    def __init__(self, input_shape, output_shape):
        super(GONEncoder, self).__init__()
        self.z_mean = layers.Dense(output_shape, input_shape=(input_shape,), activation=None, name="encoder/z_mean")
        self.z_log_var = layers.Dense(output_shape, activation=None, name="encoder/z_log_var")
        self.z = Sampling(name="encoder/z")

    def call(self, inputs):
        z_mean = self.z_mean(inputs)
        z_log_var = self.z_log_var(inputs)
        z = self.z([z_mean, z_log_var])
        return z_mean, z_log_var, z


class PreTrainedEncoder(tf.keras.Model):
    """ Encoder using a pre-trained model and learning only the mean and variance layer.
    An additional dense layer can be added between the pre-trained model and the mean and variance layers.

    Parameters
    ----------
    output_shape: int
        The dimensionality of the latent representation
    pre_trained_model: tf.keras.Model
        A pre-trained model which will be used as feature extractor
    use_dense: bool, optional
        If True, add a fully connected layer after the pre-trained model. Default True
    """

    def __init__(self, output_shape, pre_trained_model, use_dense=True):
        super(PreTrainedEncoder, self).__init__()
        self.pre_trained = pre_trained_model
        # Ensure that the pre-trained model will not be retrained
        self.pre_trained.trainable = False
        self.pre_trained.summary(print_fn=logger.debug)
        self.flatten = layers.Flatten()
        if use_dense:
            self.dense = layers.Dense(256, name="encoder/dense")
        self.z_mean = layers.Dense(output_shape, name="encoder/z_mean", kernel_initializer="zeros")
        self.z_log_var = layers.Dense(output_shape, name="encoder/z_log_var", kernel_initializer="zeros")
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.pre_trained(inputs, training=False)
        # If we use a custom classifier, all the layers output are return but this is not the case for other pretrained
        # models, thus we need to handle both cases differently
        x1 = self.flatten(x[-1])
        if hasattr(self, 'dense'):
            logger.debug("Using additional dense layer.")
            x1 = self.dense(x1)
        z_mean = self.z_mean(x1)
        z_log_var = self.z_log_var(x1)
        z = self.sampling([z_mean, z_log_var])
        return (*x, x1, z_mean, z_log_var, z)


def load_pre_trained_classifier(model_path, input_shape, n_layers=None):
    """ Load a pre-trained classifier. All the layers used for classification should contain `output` in their name
    to be removed before plugging the model to a mean and variance layer. If this is not the case, they will be kept
    when creating the encoder and this may worsen the performances.

    Parameters
    ----------
    model_path : str
        Path to the trained classifier
    input_shape : tuple or list
        The shape of the input used for the pretrained model
    n_layers: int or None
        The index of the last layer to select, can be positive or negative, similarity to Python slicing.
        If None, selects everything except the output layers.

    Returns
    -------
    tensorflow.keras.model
        The loaded classifier
    """
    model = keras.models.load_model(model_path).clf
    # Remove the output layers of the pre-trained classifier
    layers_to_add = [l.name for l in model.layers if "output" not in l.name]
    layers_to_add = layers_to_add[:n_layers]

    inputs = keras.Input(shape=input_shape)
    outputs = []
    prev_output = inputs

    for l in layers_to_add:
        logger.debug("Adding pre-trained layer {} to the model".format(l))
        prev_output = model.get_layer(l)(prev_output)
        outputs.append(prev_output)

    return keras.Model(inputs=inputs, outputs=outputs, name="pretrained_model")


def load_external_classifier(model, n_layers=None):
    """ Create a truncated version of an external classifier where the intermediate activation values are exposed.

    Parameters
    ----------
    model: tensorflow.keras.model
        The loaded classifier
    n_layers: int or None
        The index of the last layer to select, can be positive or negative, similarity to Python slicing.
        If None, selects everything except the output and last dense layer.

    Returns
    -------
    tensorflow.keras.model
        The classifier where the activation until n_layers is exposed
    """
    outputs = [layer.output for layer in model.layers[:n_layers]]
    return keras.Model(inputs=model.inputs, outputs=outputs, name="pretrained_model")
