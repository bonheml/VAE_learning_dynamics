from glob import glob
from pathlib import Path
import numpy as np
from vae_ld.data.util import natural_sort
import tensorflow as tf

from vae_ld.learning_dynamics import logger


def get_model_filename(model, model_name, param_value):
    """ Generate the filename of a model given its path, name, and parameter value.

    Parameters
    ----------
    model : str
        The path to a model saved with tf.keras.save_model. The folder containing the model checkpoint must be the seed
        number used.
    model_name : str
        The name of the model (e.g., "beta_vae")
    param_value : int, float or str
        The value of the parameter (e.g., 2 for beta_vae with beta=2)

    Returns
    -------
    str
        The generated filename

    Examples
    --------
    >>> m_path = "beta_vae/1/0/checkpoint/epoch_1090/checkpoint/epoch_10"
    >>> get_model_filename(m_path, "beta_vae", 2)
        "beta_vae_2_seed_0_epoch_10"
    """
    m_seed = model.partition("checkpoint")[0].strip("/").split("/")[-1]
    m_short = Path(model).name
    return "{}_{}_seed_{}_{}".format(model_name, param_value, m_seed, m_short)


def get_file_list(model_path, keep_n=0, selection_type="even"):
    """ Select files from those present in `model_path`.

    Parameters
    ----------
    model_path : str
        The path containing the files to select
    keep_n : int or list optional
        How many files to select. If 0, select all files. If selection type is "custom", this a list of substrings
        corresponding to the files to keep. Default 0
    selection_type : str, optional
        Can be "even" or "first". If "first", select the first `n` files. Otherwise, select evenly spaced indexes from
        the file list. Default "even".

    Returns
    -------
    list
        A list of file names
    """
    logger.debug("Retrieving files using {} selection strategy".format(selection_type))
    m_files = glob(model_path)
    m_files.sort(key=natural_sort)
    logger.debug("keep_n value is {}".format(keep_n))
    if selection_type == "custom" and type(keep_n) == list:
        logger.debug("Retrieving files containing any of {}".format(keep_n))
        m_files = [f for f in m_files for e in keep_n if e in f]
    elif selection_type == "even" and keep_n > 0:
        logger.debug("Retrieving {} files evenly distributed".format(keep_n))
        to_keep = np.linspace(0, len(m_files) - 1, num=keep_n, dtype=np.int32)
        m_files = [m_files[i] for i in to_keep]
    elif selection_type == "first" and keep_n > 0:
        logger.debug("Retrieving the first {} files".format(keep_n))
        return m_files[:keep_n]
    logger.debug("The files selected are {}".format(m_files))
    return m_files


def get_model_epoch(model):
    mstem = Path(model).stem
    i = mstem.find("epoch_")
    epoch = int(mstem[i:].split("_")[1])
    return epoch


def get_activations(data, model_path):
    """ Load a model and generate a dictionary of the activations obtained from `data`.
    We assume that the activations of each layers are exposed.

    Parameters
    ----------
    data : np.array
        A (n_examples, n_features) data matrix
    model_path : str
        The path of the model to load

    Returns
    -------
    tuple
        A tuple containing the loaded model, list of activations, and list of layer names.
    """
    model = tf.keras.models.load_model(model_path)
    if hasattr(model, "encoder"):
        acts = (data,) + model.encoder.predict(data)
        acts += model.decoder.predict(acts[-1])
        layer_names = ["input"] + [l.name for l in model.encoder.layers]
        layer_names += [l.name for l in model.decoder.layers]
    elif hasattr(model, "clf"):
        acts = [data] + list(model.clf.predict(data))
        # Flatten the list of activations from to the classification layers
        acts += acts.pop()
        acts = tuple(acts)
        layer_names = ["input"] + [l.name for l in model.clf.layers]
    else:
        raise NotImplementedError("Unknown model type. The model should contain either an encoder and decoder "
                                  "or a classifier named clf.")
    return model, acts, layer_names


def get_weights(model_path):
    """ Load a model and generate a dictionary of the weights of each layer.

    Parameters
    ----------
    model_path : str
        The path of the model to load

    Returns
    -------
    dict
        A dictionary of the form {layer_name: layer_weights}
    """
    model = tf.keras.models.load_model(model_path)
    layer_weights = {l.name: l.get_weights() for l in model.encoder.layers}
    layer_weights += {l.name: l.get_weights() for l in model.decoder.layers}
    return layer_weights


def prepare_activations(x):
    """ Flatten the activation values to get a 2D array and values very close to 0 (e.g., 1e-15) to 0.

    Parameters
    ----------
    x : tf.tensor or np.array
        A (n_example, n_features) matrix of activations

    Returns
    -------
    A (n_example, n_features) tensor of activations. If len(n_features) was initially greater than 1,
    n_features = np.prod(n_features).

    Examples
    --------
    >>> A = np.random.random((1000, 2, 2, 2))
    >>> B = prepare_activations(A)
    >>> B.shape
        (1000, 8)
    >>> C = np.array([[1.e-15, 1.e-4], [4.2, 2.e-8]])
    >>> prepare_activations(C)
        array([[0, 1e-4],
               [4.2, 0]])
    """
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    # Prevent very tiny values from causing underflow in similarity metrics later on
    x[abs(x) < 1.e-7] = 0.
    return x