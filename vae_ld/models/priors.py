import tensorflow as tf
from tensorflow.keras import layers
from vae_ld.models import logger


class Fixed(layers.Layer):
    def __init__(self, units=32, value=0):
        super().__init__()
        self.units = units
        self.value = value

    def build(self, input_shape):
        self.w = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=self.value),
            shape=(input_shape[-1], self.units),
            trainable=False,
            name="fixed"
        )

    def call(self, inputs):
        return tf.tile([self.w[0]], [tf.shape(inputs)[0], 1])


class iVAELearnedPrior(tf.keras.Model):
    def __init__(self, in_shape, output_shape):
        logger.debug("Expected input shape is {}".format(in_shape))
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = output_shape
        self.p1 = layers.Dense(50, input_shape=(in_shape,), activation=layers.LeakyReLU(alpha=0.1), name="prior_var/1")
        self.p2 = layers.Dense(50, activation=layers.LeakyReLU(alpha=0.1), name="prior_var/2")
        self.p3 = layers.Dense(50, activation=layers.LeakyReLU(alpha=0.1), name="prior_var/3")
        self.p4 = layers.Dense(output_shape, name="prior_log_var")

    def get_config(self):
        return {"in_shape": self.in_shape, "output_shape": self.out_shape}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        logger.debug("Received input shape is {}".format(inputs.shape))
        x1 = self.p1(inputs)
        x2 = self.p2(x1)
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        return x4


class iVAEFixedPrior(tf.keras.Model):
    def __init__(self, in_shape, output_shape, n=0):
        logger.debug("Expected input shape is {}".format(in_shape))
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = output_shape
        self.p1 = Fixed(units=self.out_shape, value=n)

    def get_config(self):
        return {"in_shape": self.in_shape, "output_shape": self.out_shape}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        return self.p1(inputs)
