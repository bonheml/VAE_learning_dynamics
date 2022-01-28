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


class ConvolutionalEncoder(tf.keras.Model):
    """ Convolutional encoder initially used in beta-VAE [1]. Based on Locatello et al. [2] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
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
    """ Fully connected encoder initially used in beta-VAE [1]. Based on Locatello et al. [2] implementation
    (https://github.com/google-research/disentanglement_lib)

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
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
        self.latent_shape = input_shape
        self.z_mean = layers.Dense(output_shape, input_shape=(input_shape,), activation=None, name="encoder/z_mean")
        self.z_log_var = layers.Dense(output_shape, activation=None, name="encoder/z_log_var")
        self.z = Sampling(name="encoder/z")

    def call(self, inputs):
        z_mean = self.z_mean(inputs)
        z_log_var = self.z_log_var(inputs)
        z = self.z([z_mean, z_log_var])
        return z_mean, z_log_var, z