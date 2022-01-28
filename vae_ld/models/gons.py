import tensorflow as tf
from vae_ld.models import logger
from vae_ld.models.vae_utils import compute_gaussian_kl
from vae_ld.models.vaes import VAE


class VGON(VAE):
    """ Variational GON model based on [1] and the original pytorch implementation https://github.com/cwkx/GON/blob/master/Variational-GON.py

    [1] Gradient origin networks, 2021, Bond-Taylor and Willcocks
    """

    def __init__(self, *, encoder, decoder, reconstruction_loss_fn, input_shape, latent_shape, **kwargs):
        """ Model initialisation

        :param generator: the generator to use (must be initialised beforehand). It should take a sampled z as input
        :dip_type decoder:  tf.keras.Model
        :param reconstruction_loss_fn: The metric to use for the reconstruction loss (e.g. l2, bernoulli, etc.)
        :param optimizer: The optimizer to use (e.g., adam). It must be initialised beforehand
        """
        super(VGON, self).__init__(encoder=encoder, decoder=decoder, reconstruction_loss_fn=reconstruction_loss_fn,
                                   input_shape=[input_shape], latent_shape=latent_shape, **kwargs)
        self.latent_shape = latent_shape

    # This decorator is needed to prevent input shape errors
    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def call(self, inputs):
        z = self.encoder(inputs)[-1]
        return self.decoder(z)[-1]

    # noinspection PyMethodOverriding
    def get_gradient_step_output(self, inputs, data, training=True):
        losses = {}
        z_mean, z_log_var, z = self.encoder(inputs, training=training)[-3:]
        reconstruction = self.decoder(z, training=training)[-1]
        losses["reconstruction_loss"] = tf.reduce_mean(self.reconstruction_loss_fn(data, reconstruction))
        losses["kl_loss"] = tf.reduce_mean(compute_gaussian_kl(z_log_var, z_mean))
        losses["elbo_loss"] = -tf.add(losses["reconstruction_loss"], losses["kl_loss"])
        losses["model_loss"] = self.compute_model_loss(losses["reconstruction_loss"], losses["kl_loss"], z_mean, z_log_var, z)
        return losses

    def update_metrics(self, losses):
        for m in self.metrics:
            m.update_state(losses[m.name])
        loss_res = {m.name: m.result() for m in self.metrics}
        return loss_res

    def backprop_z_gon(self, data):
        with tf.GradientTape() as inner_tape:
            z_0 = tf.zeros(data.shape[0], self.latent_shape)
            inner_tape.watch(z_0)
            inner_losses = self.get_gradient_step_output(z_0, data, training=False)
        z_gon = -inner_tape.gradient(inner_losses["model_loss"], z_0)
        return z_gon

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        logger.debug("Receive batch of size {} for training".format(data.shape))

        with tf.GradientTape() as tape:
            # Compute the the gradient of z_0, note that this is NOT the same z as the z sampled by VAEs
            # It is given as input to the mean and log variance layers
            z_gon = self.backprop_z_gon(data)
            # Do a second pass with the gradient of z_0 given as input to our mean and log variance layers
            losses = self.get_gradient_step_output(z_gon, data)

        # Update the model given the new loss
        grads = tape.gradient(losses["model_loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self.update_metrics(losses)

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        logger.debug("Receive batch of size {} for testing".format(data.shape))
        z_gon = self.backprop_z_gon(data)
        losses = self.get_gradient_step_output(z_gon, data, training=False)
        return self.update_metrics(losses)