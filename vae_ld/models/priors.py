import tensorflow as tf
from tensorflow.keras import layers
from vae_ld.models import logger


class iVAEVariancePrior(tf.keras.Model):
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