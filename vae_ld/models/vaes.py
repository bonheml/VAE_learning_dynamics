import tensorflow as tf
from tensorflow import math as tfm
from vae_ld.models.vae_utils import compute_gaussian_kl, compute_batch_tc, compute_covariance, shuffle_z


class VAE(tf.keras.Model):
    """ Vanilla VAE model based on Locatello et al. [1] implementation (https://github.com/google-research/
    disentanglement_lib) and Keras example (https://keras.io/examples/generative/vae/)

    [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """
    def __init__(self, *, encoder, decoder, reconstruction_loss_fn, save_activations, **kwargs):
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
        self.encoder = encoder.build()
        self.decoder = decoder.build()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.save_activations = save_activations

    # noinspection PyMethodMayBeStatic
    def compute_model_loss(self, reconstruction_loss, kl_loss, z_mean, z_log_var, z):
        """ Compute the model custom loss.

        :param reconstruction_loss: the reconstruction error
        :param kl_loss: the kl divergence
        :param z_mean: the mean of the latent representations
        :param z_log_var: the log variance of the latent representations
        :param z: the sampled latent representations
        :return: the model loss
        """
        return tf.add(reconstruction_loss, kl_loss, name="model_loss")

    def get_train_step_output(self, data):
        out = self.encoder(data, training=True)
        z_mean, z_log_var, z = out[-3:] if self.save_activations else out
        rec = self.decoder(z, training=True)
        reconstruction = rec[-1] if self.save_activations is True else rec

        return z_mean, z_log_var, z, reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstruction = self.get_train_step_output(data)
            reconstruction_loss = tf.reduce_mean(self.reconstruction_loss_fn(data, reconstruction),
                                                 name="reconstruction_loss")
            kl_loss = tfm.reduce_mean(compute_gaussian_kl(z_log_var, z_mean), name="kl_loss")
            elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
            model_loss = self.compute_model_loss(reconstruction_loss, kl_loss, z_mean, z_log_var, z)

        grads = tape.gradient(model_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "model_loss": model_loss,
            "elbo": -elbo,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


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
        return tf.add(reconstruction_loss, reg_kl_loss, name="model_loss")


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
        return tf.add(reconstruction_loss, reg_kl_loss, name="model_loss")


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
        return tf.add(reconstruction_loss, reg_kl_loss, name="model_loss")


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

    def compute_tc_and_discriminator_loss(self, z):
        """ Compute the loss of the discriminator and the tc loss based on Locatello et al. implementation
        (https://github.com/google-research/disentanglement_lib)

        :param z: the sampled latent representation
        :return: tuple containing tc_loss and discriminator_loss
        """
        z_shuffled = shuffle_z(z)
        discr_output = self.discriminator(z, training=True)
        logits_z, p_z = discr_output[-2:] if self.save_activations else discr_output
        discr_shuffled = self.discriminator(z_shuffled, training=True)
        _, p_z_shuffled = discr_shuffled[-2:] if self.save_activations else discr_shuffled
        # tc_loss = E[log(p_z_real) - log(p_z_fake)] = E[logits_z_real - logits_z_fake]
        tc_loss = tf.reduce_mean(logits_z[:, 0] - logits_z[:, 1], axis=0)

        # discriminator_loss = -0.5 * (E[log(p_z_real)] + E[log(p_z_shuffled_fake)])
        e_log_p_z_real = tf.reduce_mean(tfm.log(p_z[:, 0]))
        e_log_p_z_shuffled_fake = tf.reduce_mean(tfm.log(p_z_shuffled[:, 1]))
        discriminator_loss = -tf.add(0.5 * e_log_p_z_real, 0.5 * e_log_p_z_shuffled_fake, name="discriminator_loss")

        return tc_loss, discriminator_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape(persistent=True) as tape:
            z_mean, z_log_var, z, reconstruction = self.get_train_step_output(data)
            tc_loss, discriminator_loss = self.compute_tc_and_discriminator_loss(z)
            reconstruction_loss = tf.reduce_mean(self.reconstruction_loss_fn(data, reconstruction),
                                                 name="reconstruction_loss")
            kl_loss = compute_gaussian_kl(z_log_var, z_mean)
            elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
            model_loss = tf.add(elbo, self.gamma * tc_loss, name="model_loss")

        vae_trainable_weights = self.encoder.trainable_weights + self.decoder.trainable_weights
        grads_vae = tape.gradient(model_loss, vae_trainable_weights)
        self.optimizer.apply_gradients(zip(grads_vae, vae_trainable_weights))

        grads_discriminator = tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(grads_discriminator, self.discriminator.trainable_weights))
        return {
            "model_loss": model_loss,
            "elbo": -elbo,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "tc_loss": tc_loss,
            "discriminator_loss": discriminator_loss,
        }


class DIPVAE(VAE):
    def __init__(self, *, lambda_diag, lambda_off_diag, dip_type, **kwargs):
        """ Creates a DIPVAE model [1] based on Locatello et al. [2] implementation
        (https://github.com/google-research/disentanglement_lib)

        [1] Kumar, A. et al. (2018). Variational Inference of Disentangled Latent Concepts from Unlabeled Observations.
        In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada.
        [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
        Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

        :param lambda_diag: the regularisation term for diagonal values of covariance matrix
        :param lambda_off_diag: the regularisation term for off-diagonal values of covariance matrix
        :param dip_type: the type of model. Can be "i" or "ii"
        """
        super(DIPVAE, self).__init__(**kwargs)
        self.lambda_diag = lambda_diag
        self.lambda_off_diag = lambda_off_diag
        self.lambda_factor = lambda_diag * lambda_off_diag
        self.dip_type = dip_type

    def compute_dip_reg(self, cov):
        cov_diag = tf.linalg.diag_part(cov)
        cov_off_diag = cov - tf.linalg.diag(cov_diag)
        off_diag_reg = self.lambda_off_diag * tf.reduce_sum(cov_off_diag ** 2)
        diag_reg = self.lambda_factor * tf.reduce_sum((cov_diag - 1) ** 2)
        return tf.add(off_diag_reg, diag_reg)

    def compute_model_loss(self, reconstruction_loss, kl_loss, z_mean, z_log_var, z):
        if self.dip_type not in ["i", "ii"]:
            raise NotImplementedError("DIP dip_type not implemented")
        cov = compute_covariance(z_mean)
        if self.dip_type == "ii":
            cov += tf.reduce_mean(tf.linalg.diag(tf.exp(z_log_var)), axis=0)
        reg_kl_loss = self.compute_dip_reg(cov) + kl_loss
        return tf.add(reconstruction_loss, reg_kl_loss, name="model_loss")
