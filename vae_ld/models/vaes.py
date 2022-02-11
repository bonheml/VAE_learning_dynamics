import tensorflow as tf
from tensorflow import math as tfm

from vae_ld.models import logger
from vae_ld.models.vae_utils import compute_gaussian_kl, compute_batch_tc, compute_covariance, shuffle_z


class VAE(tf.keras.Model):
    """ Vanilla VAE model based on Locatello et al. [1] implementation (https://github.com/google-research/
    disentanglement_lib) and Keras example (https://keras.io/examples/generative/vae/)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """

    def __init__(self, *, encoder, decoder, reconstruction_loss_fn, input_shape, latent_shape, **kwargs):
        """ Model initialisation

        :param encoder: the encoder to use (must be initialised beforehand). It is expected to return a sampled z,
        z_mean and z_log_var
        :dip_type encoder:  tf.keras.Model
        :param decoder: the decoder to use (must be initialised beforehand). It should take a sampled z as input
        :dip_type decoder:  tf.keras.Model
        :param reconstruction_loss_fn: The metric to use for the reconstruction loss (e.g. l2, bernoulli, etc.)
        :param optimizer: The optimizer to use (e.g., adam). It must be initialised beforehand
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder.build((None, *input_shape))
        self.encoder.summary()
        self.decoder = decoder
        self.decoder.build((None, latent_shape))
        self.decoder.summary()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.elbo_loss_tracker = tf.keras.metrics.Mean(name="elbo_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.model_loss_tracker = tf.keras.metrics.Mean(name="model_loss")
        # This is needed to save the model properly
        self.built = True

    # This decorator is needed to prevent input shape errors
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, None], tf.float32)])
    def call(self, inputs):
        z = self.encoder(inputs)[-1]
        return self.decoder(z)[-1]

    @property
    def metrics(self):
        return [self.model_loss_tracker, self.elbo_loss_tracker, self.kl_loss_tracker, self.reconstruction_loss_tracker]

    def compute_model_loss(self, reconstruction_loss, kl_loss, z_mean, z_log_var, z):
        """ Compute the model custom loss.

        :param reconstruction_loss: the reconstruction error
        :param kl_loss: the kl divergence
        :param z_mean: the mean of the latent representations
        :param z_log_var: the log variance of the latent representations
        :param z: the sampled latent representations
        :return: the model loss
        """
        return tf.add(reconstruction_loss, kl_loss)

    def get_gradient_step_output(self, data, training=True):
        losses = {}
        z_mean, z_log_var, z = self.encoder(data, training=training)[-3:]
        reconstruction = self.decoder(z, training=training)[-1]
        losses["reconstruction_loss"] = tf.reduce_mean(self.reconstruction_loss_fn(data, reconstruction))
        losses["kl_loss"] = tf.reduce_mean(compute_gaussian_kl(z_log_var, z_mean))
        losses["elbo_loss"] = -tf.add(losses["reconstruction_loss"], losses["kl_loss"])
        losses["model_loss"] = self.compute_model_loss(losses["reconstruction_loss"], losses["kl_loss"], z_mean,
                                                       z_log_var, z)
        return losses

    def update_metrics(self, losses):
        for m in self.metrics:
            m.update_state(losses[m.name])
        loss_res = {m.name: m.result() for m in self.metrics}
        return loss_res

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        logger.debug("Receive batch of size {} for training".format(data.shape))
        with tf.GradientTape() as tape:
            losses = self.get_gradient_step_output(data)

        grads = tape.gradient(losses["model_loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self.update_metrics(losses)

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        logger.debug("Receive batch of size {} for testing".format(data.shape))
        losses = self.get_gradient_step_output(data, training=False)
        return self.update_metrics(losses)


class BetaVAE(VAE):
    def __init__(self, *, beta, **kwargs):
        """ Creates a beta-VAE model [1] based on Locatello et al. [2] implementation
        (https://github.com/google-research/disentanglement_lib)

        [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
        In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
        [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
        Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
        on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
        pp. 4114–4124.

        :param beta: the regularisation value
        """
        super(BetaVAE, self).__init__(**kwargs)
        self.beta = beta

    def compute_model_loss(self, reconstruction_loss, kl_loss, z_mean, z_log_var, z):
        reg_kl_loss = self.beta * kl_loss
        return tf.add(reconstruction_loss, reg_kl_loss)


class AnnealedVAE(VAE):
    def __init__(self, *, gamma, max_capacity, iteration_threshold, **kwargs):
        """ Creates a Annealed-VAE model [1] based on Locatello et al. [2] implementation
        (https://github.com/google-research/disentanglement_lib)

        [1] Burgess, C. P. et al. (2018). Understanding Disentangling in β-VAE. arXiv e-prints, 1804.03599.
        [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
        Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

        :param gamma: the regularisation value
        :param max_capacity: the maximum encoding capacity
        :param iteration_threshold: number of iterations to perform before reaching max_capacity
        """
        super(AnnealedVAE, self).__init__(**kwargs)
        self.gamma = gamma
        self.max_capacity = max_capacity * 1.
        self.iteration_threshold = iteration_threshold

    def compute_model_loss(self, reconstruction_loss, kl_loss, z_mean, z_log_var, z):
        current_step = tf.cast(self.optimizer.iterations, dtype=tf.float32)
        current_capacity = self.max_capacity * current_step / self.iteration_threshold
        c = tf.minimum(self.max_capacity, current_capacity)
        reg_kl_loss = self.gamma * tf.abs(kl_loss - c)
        return tf.add(reconstruction_loss, reg_kl_loss)


class BetaTCVAE(VAE):
    def __init__(self, *, beta, **kwargs):
        """ Creates a Beta-TCVAE model [1] based on Locatello et al. [2] implementation
        (https://github.com/google-research/disentanglement_lib)

        [1] Chen, R. T. Q. et al. (2018). Isolating Sources of Disentanglement in Variational Autoencoders.
        In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi and R. Garnett, eds.,
        Advances in Neural Information Processing Systems 31, Curran Associates, Inc., pp. 2610–2620.
        [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
        Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

        :param beta: the regularisation value
        """
        super(BetaTCVAE, self).__init__(**kwargs)
        self.beta = beta

    def compute_model_loss(self, reconstruction_loss, kl_loss, z_mean, z_log_var, z):
        tc = (self.beta - 1.) * compute_batch_tc(z, z_mean, z_log_var)
        reg_kl_loss = tc + kl_loss
        return tf.add(reconstruction_loss, reg_kl_loss)


class FactorVAE(VAE):
    def __init__(self, *, gamma, discriminator, discriminator_optimizer, **kwargs):
        """ Creates a FactorVAE model [1] based on Locatello et al. [2] implementation
        (https://github.com/google-research/disentanglement_lib)

        [1] Kim, H. and Mnih, A. (2018). Disentangling by Factorising. In J. Dy and A. Krause, eds., Proceedings of the
        35th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 80,
        Stockholmsmässan, Stockholm Sweden: PMLR, pp. 2649–2658.
        [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
        Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

        :param gamma: the regularisation value
        :param discriminator: the discriminator model
        :param discriminator_optimizer: the optimizer used for the discriminator
        """
        super(FactorVAE, self).__init__(**kwargs)
        self.gamma = gamma
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.tc_loss_tracker = tf.keras.metrics.Mean(name="tc_loss")

    @property
    def metrics(self):
        return [self.model_loss_tracker, self.elbo_loss_tracker, self.kl_loss_tracker, self.reconstruction_loss_tracker,
                self.discriminator_loss_tracker, self.tc_loss_tracker]

    def compute_tc_and_discriminator_loss(self, z, training=True):
        """ Compute the loss of the discriminator and the tc loss based on Locatello et al. implementation
        (https://github.com/google-research/disentanglement_lib)

        :param z: the sampled latent representation
        :return: tuple containing tc_loss and discriminator_loss
        """
        losses = {}
        z_shuffled = shuffle_z(z)
        logits_z, p_z = self.discriminator(z, training=training)[-2:]
        p_z_shuffled = self.discriminator(z_shuffled, training=training)[-1]
        # tc_loss = E[log(p_z_real) - log(p_z_fake)] = E[logits_z_real - logits_z_fake]
        losses["tc_loss"] = tf.reduce_mean(logits_z[:, 0] - logits_z[:, 1], axis=0)

        # discriminator_loss = -0.5 * (E[log(p_z_real)] + E[log(p_z_shuffled_fake)])
        e_log_p_z_real = tf.reduce_mean(tfm.log(p_z[:, 0]))
        e_log_p_z_shuffled_fake = tf.reduce_mean(tfm.log(p_z_shuffled[:, 1]))
        losses["discriminator_loss"] = -tf.add(0.5 * e_log_p_z_real, 0.5 * e_log_p_z_shuffled_fake)

        return losses["tc_loss"], losses["discriminator_loss"]

    def get_gradient_step_output(self, data, training=True):
        losses = {}
        z_mean, z_log_var, z = self.encoder(data, training=training)[-3:]
        reconstruction = self.decoder(z, training=training)[-1]
        losses["reconstruction_loss"] = tf.reduce_mean(self.reconstruction_loss_fn(data, reconstruction))
        losses["kl_loss"] = tf.reduce_mean(compute_gaussian_kl(z_log_var, z_mean))
        losses["elbo_loss"] = -tf.add(losses["reconstruction_loss"], losses["kl_loss"])
        losses.update(self.compute_tc_and_discriminator_loss(z, training=training))
        losses["model_loss"] = tf.add(losses["elbo_loss"], self.gamma * losses["tc_loss"])
        return losses

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape(persistent=True) as tape:
            losses = self.get_gradient_step_output(data)

        # Backprop model loss to encoder and decoder
        vae_trainable_weights = self.encoder.trainable_weights + self.decoder.trainable_weights
        grads_vae = tape.gradient(losses["model_loss"], vae_trainable_weights)
        self.optimizer.apply_gradients(zip(grads_vae, vae_trainable_weights))

        # Backprop discriminator loss separately to the discriminator only
        grads_discriminator = tape.gradient(losses["discriminator_loss"], self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(grads_discriminator, self.discriminator.trainable_weights))

        return self.update_metrics(losses)


class DIPVAE(VAE):
    def __init__(self, *, lambda_off_diag, lambda_factor, dip_type, **kwargs):
        """ Creates a DIPVAE model [1] based on Locatello et al. [2] implementation
        (https://github.com/google-research/disentanglement_lib)

        [1] Kumar, A. et al. (2018). Variational Inference of Disentangled Latent Concepts from Unlabeled Observations.
        In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada.
        [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
        Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

        :param lambda_diag: the regularisation term for diagonal values of covariance matrix
        :param lambda_factor: the regularisation term for off-diagonal values of covariance matrix
        :param dip_type: the type of model. Can be "i" or "ii"
        """
        super(DIPVAE, self).__init__(**kwargs)
        self.lambda_off_diag = lambda_off_diag
        self.lambda_diag = lambda_factor * self.lambda_off_diag
        self.dip_type = dip_type

    def compute_dip_reg(self, cov):
        cov_diag = tf.linalg.diag_part(cov)
        cov_off_diag = cov - tf.linalg.diag(cov_diag)
        off_diag_reg = self.lambda_off_diag * tf.reduce_sum(cov_off_diag ** 2)
        diag_reg = self.lambda_diag * tf.reduce_sum((cov_diag - 1) ** 2)
        return tf.add(off_diag_reg, diag_reg)

    def compute_model_loss(self, reconstruction_loss, kl_loss, z_mean, z_log_var, z):
        if self.dip_type not in ["i", "ii"]:
            raise NotImplementedError("DIP VAE {} does not exist".format(self.dip_type))
        cov = compute_covariance(z_mean)
        if self.dip_type == "ii":
            cov += tf.reduce_mean(tf.linalg.diag(tf.exp(z_log_var)), axis=0)
        reg_kl_loss = self.compute_dip_reg(cov) + kl_loss
        return tf.add(reconstruction_loss, reg_kl_loss)
