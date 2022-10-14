import tensorflow as tf
from tensorflow.python.keras import layers

from vae_ld.models import logger
from vae_ld.models.decoders import DeconvolutionalDecoder, FullyConnectedPriorDecoder
from vae_ld.models.divergences import KLD
from vae_ld.models.encoders import ConvolutionalEncoder, FullyConnectedPriorEncoder
from vae_ld.models.losses import BernoulliLoss
from vae_ld.models.vae_utils import compute_batch_tc, compute_covariance


class GenericVAE(tf.keras.Model):
    """ Vanilla VAE model based on Locatello et al. [1]
    `implementation <https://github.com/google-research/disentanglement_lib>`_
    and `Keras example <https://keras.io/examples/generative/vae/>`_.

    Parameters
    ----------
    encoder : tf.keras.Model or None, optional
        The encoder. Expected to last three activation matrices should be z_mean, z_log_var, and sampled z.
        If None, a convolutional encoder with the architecture used in beta vae will be created
    decoder : tf.keras.Model or None, optional
        The decoder. Takes a sampled z as input
        If None, a convolutional decoder with the architecture used in beta vae will be created
    reconstruction_loss_fn : function, optional
        The metric to use for the reconstruction loss (e.g. l2, bernoulli)
        Default is BernoulliLoss
    input_shape : list or tuple
        The shape of the input given to the encoder.
        Default is (64,64,3)
    latent_shape : int
        The number of dimensions of z.
        Default is 10

    References
    ----------
    .. [1] Locatello et al, (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
           Representations. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:4114-4124

    Note
    -----
    The encoder and decoder are assumed to be initialised beforehand.
    """

    def __init__(self, *args, encoder=None, decoder=None, reconstruction_loss_fn=BernoulliLoss(),
                 regularisation_loss_fn=KLD(), input_shape=(64, 64, 3), latent_shape=10, output_shape=(64, 64, 3),
                 **kwargs):
        self.n_samples = kwargs.pop("n_samples", 1)
        if self.n_samples <= 0:
            raise ValueError("The number of samples must be greater than 0.")
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.regularisation_loss_fn = regularisation_loss_fn
        self.regularisation_loss_tracker = tf.keras.metrics.Mean(name="regularisation_loss")
        self.elbo_loss_tracker = tf.keras.metrics.Mean(name="elbo_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.model_loss_tracker = tf.keras.metrics.Mean(name="model_loss")
        self.latent_shape = latent_shape
        self.in_shape = input_shape
        self.out_shape = output_shape

    @property
    def metrics(self):
        return [self.model_loss_tracker, self.elbo_loss_tracker, self.regularisation_loss_tracker, self.reconstruction_loss_tracker]

    def compute_model_loss(self, *args, **kwargs):
        """ Compute the loss specific to the learning objective used.

        Returns
        -------
        tf.Tensor
            The model loss
        """
        rec_loss, reg_loss = kwargs.get("reconstruction_loss"), kwargs.get("regularisation_loss")
        return tf.add(rec_loss, reg_loss)

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
        losses["model_loss"] = self.compute_model_loss(losses["reconstruction_loss"], losses["regularisation_loss"],
                                                       z_mean, z_log_var, z)
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
        logger.debug("Receive batch of size {} for training".format(data.shape if not isinstance(data, tuple) else
                                                                    [d.shape for d in data]))
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
        logger.debug("Receive batch of size {} for testing".format(data.shape if not isinstance(data, tuple) else
                                                                   [d.shape for d in data]))
        losses = self.get_gradient_step_output(data, training=False)
        return self.update_metrics(losses)


class VAE(GenericVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.encoder is None:
            self.encoder = ConvolutionalEncoder(self.in_shape, self.latent_shape)
        self.encoder.build((None, *self.in_shape))
        self.encoder.summary(print_fn=logger.info)

        if self.decoder is None:
            self.decoder = DeconvolutionalDecoder(self.latent_shape, self.out_shape)
        self.decoder.build((None, self.latent_shape * self.n_samples))
        self.decoder.summary(print_fn=logger.info)

        self.built = True

    # This decorator is needed to prevent input shape errors
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, None], tf.float32)])
    def call(self, inputs):
        z = self.encoder(inputs)[-1]
        return self.decoder(z)[-1]


class MultiInputVAE(GenericVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.encoder is None:
            self.encoder = ConvolutionalEncoder(self.in_shape[0], self.latent_shape)
        # *[...], or *(...), cast a comprehension list to a tuple
        self.encoder.build([(None, *i) for i in self.in_shape])
        self.encoder.summary(print_fn=logger.info)

        if self.decoder is None:
            self.decoder = DeconvolutionalDecoder(self.latent_shape, self.out_shape)
        self.decoder.build((None, self.latent_shape * self.n_samples))
        self.decoder.summary(print_fn=logger.info)

        self.built = True

    # This decorator is needed to prevent input shape errors
    @tf.function(input_signature=[(tf.TensorSpec([None, None, None, None], tf.float32),
                                  tf.TensorSpec([None, None], tf.float32)),])
    def call(self, inputs):
        z = self.encoder(inputs)[-1]
        return self.decoder(z)[-1]


class IVAE(MultiInputVAE):
    """ Creates an iVAE model based on Khemakhem et al. [1]
    `implementation <https://github.com/siamakz/iVAE/blob/master/lib/models.py>`_.

    Parameters
    ----------
    prior_model: tf.keras.Model or None, optional
        The intialised conditional prior model. If None, creates a model with the same architecture as [2].
    prior_mean: int, optional
        The mean of the conditional prior. Default is 0 as in [1].
    prior_shape: int, optional
        Number of latent dimensions for the variance. Default is 10.

    References
    ----------
    .. [1] Khemakhem, I., Kingma, D., Monti, R., & Hyvarinen, A. (2020, June). Variational autoencoders
           and nonlinear ica: A unifying framework. In International Conference on Artificial Intelligence
           and Statistics (pp. 2207-2217). PMLR.
    .. [2] Mita, G., Filippone, M., & Michiardi, P. (2021, July). An identifiable double vae for disentangled
           representations. In International Conference on Machine Learning (pp. 7769-7779). PMLR.
    """
    def __init__(self, *args, prior_model=None, prior_mean=0, prior_shape=10, **kwargs):
        input_shape = [list(kwargs.pop("input_shape")), [prior_shape,]]
        super().__init__(*args, input_shape=input_shape, **kwargs)
        self.prior_mean = prior_mean
        self.prior_model = prior_model
        if self.prior_model is None:
            self.prior_model = tf.keras.Sequential(name="prior")
            self.prior_model.add(layers.Dense(1000, input_shape=(prior_shape,), activation=layers.LeakyReLU(alpha=0.2),
                                              name="prior/1"))
            for i in range(2):
                self.prior_model.add(layers.Dense(1000, activation=layers.LeakyReLU(alpha=0.2),
                                                  name="prior/{}".format(i+1)))
            self.prior_model.add(layers.Dense(self.latent_shape, name="prior/log_var"))
        self.prior_model.build((None, prior_shape))
        self.prior_model.summary(print_fn=logger.info)

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
        # Estimate the log var of p_{\lambda, T}(z|u). The mean is fixed, as in [1].
        prior_log_var = self.prior_model(data[1], training=training)
        prior_mean = self.prior_mean * tf.ones_like(prior_log_var)

        # Compute E_q_{\phi}(z|x,u)[log p_f(x|z)] - KL(q_{\phi}(z|x,u) || p_{\lambda, T}(z|u))
        losses["reconstruction_loss"] = tf.reduce_mean(self.reconstruction_loss_fn(data[0], reconstruction))
        losses["regularisation_loss"] = tf.reduce_mean(self.regularisation_loss_fn(z_log_var, z_mean,
                                                                                   z2_log_var=prior_log_var,
                                                                                   z2_mean=prior_mean))
        losses["elbo_loss"] = -tf.add(losses["reconstruction_loss"], losses["regularisation_loss"])
        losses["model_loss"] = self.compute_model_loss(**losses)
        return losses


class IDVAE(MultiInputVAE):
    """ Creates an IDVAE model based on Mita et al. [1]
        `implementation <https://github.com/grazianomita/disentanglement_idvae/blob/main/disentanglement/models/idvae.py>`_.

        Parameters
        ----------
        prior_encoder: tf.keras.Model or None, optional
            The intialised conditional prior encoder. If None, creates a model with the same architecture as [1].
        prior_decoder: tf.keras.Model or None, optional
            The intialised conditional prior decoder. If None, creates a model with the same architecture as [1].
        prior_shape: int, optional
            Number of latent dimensions of the prior model. Default is 10.

        References
        ----------
        .. [1] Mita, G., Filippone, M., & Michiardi, P. (2021, July). An identifiable double vae for disentangled
               representations. In International Conference on Machine Learning (pp. 7769-7779). PMLR.
        """

    def __init__(self, *args, prior_encoder=None, prior_decoder=None, prior_shape=10, **kwargs):
        input_shape = [list(kwargs.pop("input_shape")), [prior_shape, ]]
        super().__init__(*args, input_shape=input_shape, **kwargs)
        self.prior_encoder = prior_encoder
        self.prior_decoder = prior_decoder
        if self.prior_encoder is None:
            self.prior_encoder = FullyConnectedPriorEncoder((prior_shape,), self.latent_shape)
        if self.prior_decoder is None:
            self.prior_decoder = FullyConnectedPriorDecoder(self.latent_shape, prior_shape)
        self.prior_encoder.build((None, prior_shape))
        self.prior_encoder.summary(print_fn=logger.info)
        self.prior_decoder.build((None, self.latent_shape * self.n_samples))
        self.prior_decoder.summary(print_fn=logger.info)
        self.prior_regularisation_loss_tracker = tf.keras.metrics.Mean(name="prior_regularisation_loss")
        self.prior_elbo_loss_tracker = tf.keras.metrics.Mean(name="prior_elbo_loss")
        self.prior_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="prior_reconstruction_loss")

    @property
    def metrics(self):
        return [self.model_loss_tracker, self.elbo_loss_tracker, self.regularisation_loss_tracker,
                self.reconstruction_loss_tracker, self.prior_elbo_loss_tracker, self.prior_regularisation_loss_tracker,
                self.prior_reconstruction_loss_tracker]

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
        zp_mean, zp_log_var, zp = self.prior_encoder(data[1], training=training)[-3:]
        prior_reconstruction = self.prior_decoder(zp, training=training)[-1]

        # Compute E_q_{\psi}(z|u)[log p_{\theta}(u|z)] - KL(q_{\psi}(z|u) || p_{\theta}(z))
        losses["prior_reconstruction_loss"] = tf.reduce_mean(self.reconstruction_loss_fn(data[1], prior_reconstruction))
        losses["prior_regularisation_loss"] = tf.reduce_mean(self.regularisation_loss_fn(zp_log_var, zp_mean))

        # Compute E_q_{\phi}(z|x,u)[log p_f(x|z)] - KL(q_{\phi}(z|x,u) || p_{\lambda, T}(z|u))
        losses["reconstruction_loss"] = tf.reduce_mean(self.reconstruction_loss_fn(data[0], reconstruction))
        losses["regularisation_loss"] = tf.reduce_mean(self.regularisation_loss_fn(z_log_var, z_mean,
                                                                                   z2_log_var=zp_log_var,
                                                                                   z2_mean=zp_mean))
        losses["elbo_loss"] = -tf.add(losses["reconstruction_loss"], losses["regularisation_loss"])
        losses["prior_elbo_loss"] = -tf.add(losses["prior_reconstruction_loss"], losses["prior_regularisation_loss"])
        losses["model_loss"] = self.compute_model_loss(**losses)
        return losses

    def compute_model_loss(self, *args, **kwargs):
        elbo, prior_elbo = kwargs.get("elbo_loss"), kwargs.get("prior_elbo_loss")
        return -tf.add(elbo, prior_elbo)



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
    def __init__(self, *args, beta=1, **kwargs):
        super(BetaVAE, self).__init__(*args, **kwargs)
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
    def __init__(self, *args, gamma=1, max_capacity=1, iteration_threshold=1, **kwargs):
        super(AnnealedVAE, self).__init__(*args, **kwargs)
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
    def __init__(self, *args, iteration_threshold=1, **kwargs):
        super(AnnealedVAEB, self).__init__(*args, **kwargs)
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
    def __init__(self, *args, beta=1, **kwargs):
        super(BetaTCVAE, self).__init__(*args, **kwargs)
        self.beta = beta

    def compute_model_loss(self, reconstruction_loss, regularisation_loss, z_mean, z_log_var, z):
        tc = (self.beta - 1.) * compute_batch_tc(z, z_mean, z_log_var)
        reg_regularisation_loss = tc + regularisation_loss
        return tf.add(reconstruction_loss, reg_regularisation_loss)


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
    def __init__(self, *args, lambda_off_diag=1, lambda_factor=1, dip_type="ii", **kwargs):
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
