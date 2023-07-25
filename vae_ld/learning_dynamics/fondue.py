import gc
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
    decoder = instantiate(cfg.model.decoder, in_shape=n)
    model = instantiate(cfg.model, encoder=encoder, decoder=decoder, latent_shape=n)
    model.compile(optimizer=optimizer, run_eagerly=True)
    return model


def train_model_and_get_estimate(model, sampler, estimator, data_examples, cfg):
    """ Train a model for a few steps and get the IDE of its mean and sampled representation

    Parameters
    ----------
    model:
        the model to use
    sampler:
        The data sampler
    estimator:
        The estimator to use
    data_examples:
        The data used to compute the IDE
    cfg:
        The config containing the model info

    Returns
    -------
    mean and sampled IDEs
    """
    model.fit(sampler, epochs=cfg.max_epochs, steps_per_epoch=cfg.steps_per_epoch, batch_size=cfg.batch_size)
    _, acts, _ = get_encoder_latents_activations(data_examples, None, model)
    acts = [prepare_activations(act) for act in acts]
    if estimator.__self__.__class__.__name__ == "InformationBottleneck":
        X_hat = prepare_activations(model.decoder.predict(acts[-1])[-1])
        mean = estimator(data_examples, X_hat)
    else:
        mean = estimator(acts[0])
    sampled = estimator(acts[-1])
    return mean, sampled


def train_model_and_get_var_types(model, sampler, estimator, data_examples, cfg):
    """ Train a model for a few steps and get the variable types from the variance representation

    Parameters
    ----------
    model:
        the model to use
    sampler:
        The data sampler
    estimator:
        The variable filter estimator to use
    data_examples:
        The data used to compute the variable type
    cfg:
        The config containing the model info

    Returns
    -------
    number of passive, mixed and active variables
    """
    model.fit(sampler, epochs=cfg.max_epochs, steps_per_epoch=cfg.steps_per_epoch, batch_size=cfg.batch_size)
    _, acts, _ = get_encoder_latents_activations(data_examples, None, model)
    var_acts = prepare_activations(acts[1])
    var_types = estimator(var_acts)
    return var_types["active_variables"].unique()[0], var_types["mixed_variables"].unique()[0], var_types["passive_variables"].unique()[0]


def get_mem(mem, pivot, cfg, optimizer, sampler, estimator, data_examples):
    logger.info("Retrieving estimates for {} latents".format(pivot))
    res = mem.get(pivot, None)

    if res is None:
        logger.debug("Instantiate model with {} latents".format(pivot))
        model = init_model_with_n_latents(cfg, pivot, optimizer)
        logger.debug("Computing score for mean and sampled representations")
        mean, sampled = train_model_and_get_estimate(model, sampler, estimator, data_examples, cfg)
        mem[pivot] = (sampled, mean)
        del model
        gc.collect()

    return mem[pivot]


def fondue(estimator, data_ide, data_examples, sampler, cfg):
    """ FONDUE algorithm, retrieve the optimal number of latent dimensions to use for a VAE.

    Parameters
    ----------
    estimator:
        The estimator to use
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

    lower_bound, upper_bound, pivot = 0, np.inf, data_ide
    mem = {}
    threshold = cfg.threshold
    logger.debug("The threshold is {}".format(threshold))
    optimizer = instantiate(cfg.optimizer)

    while pivot != lower_bound:
        logger.debug("p: {}, l: {}, u: {}".format(pivot, lower_bound, upper_bound))
        sampled, mean = get_mem(mem, pivot, cfg, optimizer, sampler, estimator, data_examples)
        diff = sampled - mean

        logger.info("diff = {} - {} = {}".format(sampled, mean, diff))
        if diff <= threshold:
            lower_bound = pivot
            pivot = min(pivot * 2, upper_bound)
        else:
            upper_bound = pivot
            pivot = (lower_bound + upper_bound) // 2

    return pivot


def fondue_var_type(estimator, data_ide, data_examples, sampler, cfg):
    """ FONDUE algorithm, retrieve the optimal number of latent dimensions to use for a VAE using IDE.

    Parameters
    ----------
    estimator:
        The variable type estimator to use
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
    latent_dim = 2 * data_ide
    optimizer = instantiate(cfg.optimizer)
    all_active = True

    while all_active:
        logger.debug("Instantiate model with {} latents".format(latent_dim))
        model = init_model_with_n_latents(cfg, latent_dim, optimizer)
        logger.debug("Computing variable types from mean representations")
        active_vars, mixed_vars, passive_vars = train_model_and_get_var_types(model, sampler, estimator, data_examples, cfg)
        logger.debug("Found {} active variables, {} mixed variables, and {} passive variables".format(active_vars, mixed_vars, passive_vars))
        if passive_vars > 0 and cfg.threshold is True:
            return active_vars + mixed_vars
        if cfg.threshold is False and (mixed_vars > 0 or passive_vars > 0):
            return active_vars
        latent_dim *= 2
        del model
        gc.collect()
