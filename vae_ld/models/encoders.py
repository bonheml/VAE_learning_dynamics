import tensorflow as tf
from keras.models import load_model
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
    def __init__(self, input_shape, output_shape):
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
        self.z_mean = layers.Dense(output_shape, name="encoder/z_mean")
        self.z_log_var = layers.Dense(output_shape, name="encoder/z_log_var")
        self.sampling = Sampling()

    def call(self, inputs):
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
        self.flatten = layers.Flatten(name="encoder/flatten")
        if use_dense:
            self.dense = layers.Dense(256, activation="relu", name="encoder/dense")
        self.z_mean = layers.Dense(output_shape, name="encoder/z_mean")
        self.z_log_var = layers.Dense(output_shape, name="encoder/z_log_var")
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.pre_trained(inputs, training=False)
        x1 = self.flatten(x)
        if hasattr(self, 'dense'):
            x1 = self.dense(x1)
        z_mean = self.z_mean(x1)
        z_log_var = self.z_log_var(x1)
        z = self.sampling([z_mean, z_log_var])
        return (*x, x1, z_mean, z_log_var, z)


def load_pre_trained_classifier(model_path, input_shape):
    """ Load a pre-trained classifier. All the layers used for classification should contain `output` in their name
    to be removed before plugging the model to a mean and variance layer. If this is not the case, they will be kept
    when creating the encoder and this may worsen the performances.

    Parameters
    ----------
    model_path : str
        Path to the trained classifier
    input_shape : tuple or list
        The shape of the input used for the pretrained model

    Returns
    -------
    tensorflow.keras.model
        The loaded classifier
    """
    model = load_model(model_path)
    # Remove the output layers of the pre-trained classifier
    layers_to_add = [l.name for l in model.layers if "output" not in l.name]
    # Remove the fully connected layer just before the output layers
    layers_to_add.pop()

    inputs = keras.Input(shape=input_shape)
    outputs = []
    prev_output = inputs

    for l in layers_to_add:
        logger.debug("Adding pre-trained layer {} to the model".format(l))
        prev_output = model.get_layer(l)(prev_output)
        outputs.append(prev_output)

    return keras.Model(inputs=inputs, outputs=outputs, name="pretrained_model")
