from hydra.utils import instantiate
from vae_ld.learning_dynamics import logger
from vae_ld.learning_dynamics.utils import get_encoder_latents_activations, prepare_activations
import numpy as np


def init_model_with_n_latents(cfg, n, optimizer):
    """ Initialise a model given a config and a specific number of latent dimensions

    Parameters
    ----------
    cfg:
        the config containing the model info
    n: int
        the number of latent dimensions
    optimizer:
        The optimiser to use

    Returns
    -------
    The created model
    """
    encoder = instantiate(cfg.model.encoder, output_shape=n)
    decoder = instantiate(cfg.model.decoder, input_shape=n)
    model = instantiate(cfg.model, encoder=encoder, decoder=decoder, latent_shape=n)
    model.compile(optimizer=optimizer, run_eagerly=True)
    return model


def train_model_and_get_ides(model, sampler, id_estimator, data_examples, cfg):
    """ Train a model for 1 epoch and the the IDE of its mean and sampled representation

    Parameters
    ----------
    model:
        the model to use
    sampler:
        The data sampler
    id_estimator:
        The IDE estimator to use
    data_examples:
        The data used to compute the IDE
    cfg:
        The config containing the model info

    Returns
    -------
    mean and sampled IDEs
    """
    model.fit(sampler, epochs=cfg.max_epochs, steps_per_epochs=cfg.steps_per_epochs, batch_size=cfg.batch_size)
    _, acts, _ = get_encoder_latents_activations(data_examples, None, model)
    acts = [prepare_activations(act) for act in acts]
    mean_ide = id_estimator(acts[0])
    sampled_ide = id_estimator(acts[-1])
    return mean_ide, sampled_ide


def fondue(id_estimator, data_ide, data_examples, sampler, cfg):
    """ FONDUE algorithm, retrieve the optimal number of latent dimensions to use for a VAE.

    Parameters
    ----------
    id_estimator:
        The ID estimator to use
    data_ide:
        The IDE of the data
    data_examples:
        The data used to compute the IDE
    sampler:
        The data sampler
    cfg:
        The config containing the model info


    Returns
    -------
    The optimal number of latent dimensions
    """

    infimum, supremum, upper_bound = 0, np.inf, data_ide
    ides_diff = {}
    threshold = (cfg.threshold * upper_bound) / 100
    logger.debug("The threshold is {}".format(threshold))
    optimizer = instantiate(cfg.optimizer)

    while upper_bound != infimum and upper_bound > 0:
        logger.debug("Upper bound: {}, Infimum: {}, Supremum: {}".format(upper_bound, infimum, supremum))
        diff = ides_diff.get(upper_bound, None)

        if diff is None:
            logger.debug("Instantiate model with {} latents".format(upper_bound))
            model = init_model_with_n_latents(cfg, upper_bound, optimizer)
            logger.debug("Computing IDE of mean and sampled representations")
            mean_ide, sampled_ide = train_model_and_get_ides(model, sampler, id_estimator, data_examples, cfg)
            ides_diff[upper_bound] = sampled_ide - mean_ide

        logger.debug("The difference between mean and sampled IDE is {}".format(ides_diff[upper_bound]))
        if ides_diff[upper_bound] > threshold:
            supremum = upper_bound
            infimum, upper_bound = infimum, (infimum + upper_bound) // 2
        else:
            infimum, upper_bound = upper_bound, min(upper_bound * 2, supremum)

    return upper_bound
