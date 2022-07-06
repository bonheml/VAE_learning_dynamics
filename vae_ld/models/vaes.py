import tensorflow as tf
from tensorflow import math as tfm

from vae_ld.models import logger
from vae_ld.models.vae_utils import compute_batch_tc, compute_covariance, shuffle_z


class VAE(tf.keras.Model):
    """ Vanilla VAE model based on Locatello et al. [1] `implementation <https://github.com/google-research/disentanglement_lib>`_
    and `Keras example <https://keras.io/examples/generative/vae/>`_.

    Parameters
    ----------
    encoder : tf.keras.Model
        The encoder. Expected to last three activation matrices should be z_mean, z_log_var, and sampled z.
    decoder : tf.keras.Model
        The decoder. Takes a sampled z as input
    reconstruction_loss_fn : function
        The metric to use for the reconstruction loss (e.g. l2, bernoulli)
    input_shape : list
        The shape of the input given to the encoder
    latent_shape : list
        The number of dimensions of z

    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

    Note
    -----
    The encoder and decoder are assumed to be initialised beforehand.
    """

    def __init__(self, *, encoder, decoder, reconstruction_loss_fn, regularisation_loss_fn, input_shape, latent_shape, **kwargs):
        n_samples = kwargs.pop("n_samples", 1)
        if n_samples <= 0:
            raise ValueError("The number of samples must be greater than 0.")
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder.build((None, *input_shape))
        self.encoder.summary(print_fn=logger.info)
        self.decoder = decoder
        self.decoder.build((None, latent_shape * n_samples))
        self.decoder.summary(print_fn=logger.info)
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.regularisation_loss_fn = regularisation_loss_fn
        self.regularisation_loss_tracker = tf.keras.metrics.Mean(name="regularisation_loss")
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
        return [self.model_loss_tracker, self.elbo_loss_tracker, self.regularisation_loss_tracker, self.reconstruction_loss_tracker]

    def compute_model_loss(self, reconstruction_loss, regularisation_loss, z_mean, z_log_var, z):
        """ Compute the loss specific to the learning objective used.

        Parameters
        ----------
        reconstruction_loss : tf.Tensor
            The reconstruction error
        regularisation_loss : tf.Tensor
            The kl divergence
        z_mean : tf.Tensor
            The (batch_size, latent_dim) activations of the mean layer
        z_log_var : tf.Tensor
            The (batch_size, latent_dim) activations of the log variance layer
        z : tf.Tensor
            The (batch_size, latent_dim) sampled values

        Returns
        -------
        tf.Tensor
            The model loss
        """
        return tf.add(reconstruction_loss, regularisation_loss)

    def get_gradient_step_output(self, data, training=True):
        """ Do one gardient step.

        Parameters
        ----------
        data
            The batch of data to use
        training : bool, optional
            Whether the model is training or testing.

        Returns
        -------
        list
            The update losses
        """
        losses = {}
        z_mean, z_log_var, z = self.encoder(data, training=training)[-3:]
        reconstruction = self.decoder(z, training=training)[-1]
        losses["reconstruction_loss"] = tf.reduce_mean(self.reconstruction_loss_fn(data, reconstruction))
        losses["regularisation_loss"] = tf.reduce_mean(self.regularisation_loss_fn(z_log_var, z_mean))
        losses["elbo_loss"] = -tf.add(losses["reconstruction_loss"], losses["regularisation_loss"])
        losses["model_loss"] = self.compute_model_loss(losses["reconstruction_loss"], losses["regularisation_loss"], z_mean,
                                                       z_log_var, z)
        return losses

    def update_metrics(self, losses):
        """ Update the evaluation metrics

        Parameters
        ----------
        losses : list
            The metrics to track

        Returns
        -------
        dict
            The updated metrics in the format {metric_name: metric_value}
        """
        for m in self.metrics:
            m.update_state(losses[m.name])
        loss_res = {m.name: m.result() for m in self.metrics}
        return loss_res

    def train_step(self, data):
        """ Perform a training step.

        Parameters
        ----------
        data
            The batch of data to use

        Returns
        -------
        dict
            The updated evaluation metrics in the format {metric_name: metric_value}
        """
        if isinstance(data, tuple):
            data = data[0]
        logger.debug("Receive batch of size {} for training".format(data.shape))
        with tf.GradientTape() as tape:
            losses = self.get_gradient_step_output(data)

        grads = tape.gradient(losses["model_loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self.update_metrics(losses)

    def test_step(self, data):
        """ Perform one test step.

        Parameters
        ----------
        data
            The batch of data to use

        Returns
        -------
        dict
            The updated evaluation metrics in the format {metric_name: metric_value}
        """
        if isinstance(data, tuple):
            data = data[0]
        logger.debug("Receive batch of size {} for testing".format(data.shape))
        losses = self.get_gradient_step_output(data, training=False)
        return self.update_metrics(losses)


class BetaVAE(VAE):
    """ Creates a beta-VAE model [1] based on Locatello et al. [2]
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    Parameters
    ----------
    beta : int or float
        The regularisation value

    References
    ----------
    .. [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017).
           β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In 5th International
           Conference on Learning Representations, ICLR 2017, Toulon, France.
    .. [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """
    def __init__(self, *, beta, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.beta = beta

    def compute_model_loss(self, reconstruction_loss, regularisation_loss, z_mean, z_log_var, z):
        reg_regularisation_loss = self.beta * regularisation_loss
        return tf.add(reconstruction_loss, reg_regularisation_loss)


class AnnealedVAE(VAE):
    """ Creates a Annealed-VAE model [1] based on Locatello et al. [2]
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    Parameters
    ----------
    gamma : float
        The regularisation value
    max_capacity : float
        The maximum encoding capacity
    iteration_threshold : int
        Number of iterations to perform before reaching max_capacity

    References
    ----------
    .. [1] Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2018).
           Understanding disentangling in $\beta $-VAE. arXiv preprint arXiv:1804.03599.
    .. [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """
    def __init__(self, *, gamma, max_capacity, iteration_threshold, **kwargs):
        super(AnnealedVAE, self).__init__(**kwargs)
        self.gamma = gamma
        self.max_capacity = max_capacity * 1.
        self.iteration_threshold = iteration_threshold

    def compute_model_loss(self, reconstruction_loss, regularisation_loss, z_mean, z_log_var, z):
        current_step = tf.cast(self.optimizer.iterations, dtype=tf.float32)
        current_capacity = self.max_capacity * current_step / self.iteration_threshold
        c = tf.minimum(self.max_capacity, current_capacity)
        reg_regularisation_loss = self.gamma * tf.abs(regularisation_loss - c)
        return tf.add(reconstruction_loss, reg_regularisation_loss)


class AnnealedVAEB(VAE):
    """ Creates a VAE with linear annealing as in NLP [1] where beta is linearly increased, contrary to Annealed-VAE
    from disentanglement [2], where beta is penalised over time.

    Parameters
    ----------
    gamma : float
        The amount by which beta is increased after each epoch until it reaches 1.

    iteration_threshold : int
        Number of iterations to perform before reaching max_capacity

    References
    ----------
    .. [1] Bowman, S., Vilnis, L., Vinyals, O., Dai, A., Jozefowicz, R., & Bengio, S. (2016, August).
           Generating Sentences from a Continuous Space. In Proceedings of The 20th SIGNLL Conference on Computational
           Natural Language Learning (pp. 10-21).
    .. [2] Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2018).
           Understanding disentangling in $\beta $-VAE. arXiv preprint arXiv:1804.03599.
    """
    def __init__(self, *, iteration_threshold, **kwargs):
        super(AnnealedVAEB, self).__init__(**kwargs)
        self.gamma = 0.
        self.iteration_threshold = iteration_threshold

    def compute_model_loss(self, reconstruction_loss, regularisation_loss, z_mean, z_log_var, z):
        reg_regularisation_loss = self.gamma * regularisation_loss
        # We increase gamma for the next step
        if self.gamma < 1.:
            self.gamma += min(1., 1 / self.iteration_threshold)
        return tf.add(reconstruction_loss, reg_regularisation_loss)


class BetaTCVAE(VAE):
    """ Creates a Beta-TCVAE model [1] based on Locatello et al. [2]
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    Parameters
    ----------
    beta : int or float
        The regularisation value

    References
    ----------
    .. [1] Chen, R. T., Li, X., Grosse, R. B., & Duvenaud, D. K. (2018). Isolating sources of disentanglement in
           variational autoencoders. Advances in neural information processing systems, 31.
    .. [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """
    def __init__(self, *, beta, **kwargs):
        super(BetaTCVAE, self).__init__(**kwargs)
        self.beta = beta

    def compute_model_loss(self, reconstruction_loss, regularisation_loss, z_mean, z_log_var, z):
        tc = (self.beta - 1.) * compute_batch_tc(z, z_mean, z_log_var)
        reg_regularisation_loss = tc + regularisation_loss
        return tf.add(reconstruction_loss, reg_regularisation_loss)


class FactorVAE(VAE):
    """ Creates a FactorVAE model [1] based on Locatello et al. [2]
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    Parameters
    ----------
    gamma : float
        The regularisation value
    discriminator : tf.keras.Model
        The initialised discriminator model
    discriminator_optimizer
        The optimizer used for the discriminator

    References
    ----------
    .. [1] Kim, H., & Mnih, A. (2018, July). Disentangling by factorising. In International Conference on
           Machine Learning (pp. 2649-2658). PMLR.
    .. [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """
    def __init__(self, *, gamma, discriminator, discriminator_optimizer, **kwargs):
        super(FactorVAE, self).__init__(**kwargs)
        self.gamma = gamma
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.tc_loss_tracker = tf.keras.metrics.Mean(name="tc_loss")

    @property
    def metrics(self):
        return [self.model_loss_tracker, self.elbo_loss_tracker, self.regularisation_loss_tracker, self.reconstruction_loss_tracker,
                self.discriminator_loss_tracker, self.tc_loss_tracker]

    def compute_tc_and_discriminator_loss(self, z, training=True):
        """ Compute the loss of the discriminator and the tc loss based on Locatello et al.
        `implementation <https://github.com/google-research/disentanglement_lib>`_.

        Parameters
        ----------
        z
            The sampled latent representation
        training: bool, optional
            Whether the model is training or testing.

        Returns
        -------
        tuple
            Total correlation and discriminator losses
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
        losses["regularisation_loss"] = tf.reduce_mean(self.regularisation_loss_fn(z_log_var, z_mean))
        losses["elbo_loss"] = -tf.add(losses["reconstruction_loss"], losses["regularisation_loss"])
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
    """ Creates a DIP VAE model [1] based on Locatello et al. [2]
    `implementation <https://github.com/google-research/disentanglement_lib>`_.

    Parameters
    ----------
    lambda_off_diag : int or float
        The regularisation term for the diagonal values of covariance matrix
    lambda_factor : int or float
        The regularisation term for the off-diagonal values of covariance matrix
    dip_type : str
        The type of DIP VAE loss to use. Can be "i" or "ii".

    References
    ----------
    .. [1] Kumar, A., Sattigeri, P., & Balakrishnan, A. (2018). Variational Inference of Disentangled Latent
           Concepts from Unlabeled Observations. In International Conference on Learning Representations.
    .. [2] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124
    """
    def __init__(self, *, lambda_off_diag, lambda_factor, dip_type, **kwargs):
        super(DIPVAE, self).__init__(**kwargs)
        self.lambda_off_diag = lambda_off_diag
        self.lambda_diag = lambda_factor * self.lambda_off_diag
        self.dip_type = dip_type

    def compute_dip_reg(self, cov):
        """ Compute DIP VAE loss

        Parameters
        ----------
        cov : tf.tensor
            A covariance matrix

        Returns
        -------
        tf.tensor
            DIP VAE loss

        Note
        -----
        The covariance is over `z_mean` for DIP VAE I and `z_var` for DIP VAE II
        """
        cov_diag = tf.linalg.diag_part(cov)
        cov_off_diag = cov - tf.linalg.diag(cov_diag)
        off_diag_reg = self.lambda_off_diag * tf.reduce_sum(cov_off_diag ** 2)
        diag_reg = self.lambda_diag * tf.reduce_sum((cov_diag - 1) ** 2)
        return tf.add(off_diag_reg, diag_reg)

    def compute_model_loss(self, reconstruction_loss, regularisation_loss, z_mean, z_log_var, z):
        if self.dip_type not in ["i", "ii"]:
            raise NotImplementedError("DIP VAE {} does not exist".format(self.dip_type))
        cov = compute_covariance(z_mean)
        if self.dip_type == "ii":
            cov += tf.reduce_mean(tf.linalg.diag(tf.exp(z_log_var)), axis=0)
        reg_regularisation_loss = self.compute_dip_reg(cov) + regularisation_loss
        return tf.add(reconstruction_loss, reg_regularisation_loss)
