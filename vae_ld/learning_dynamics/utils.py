from glob import glob
from pathlib import Path
import numpy as np
from vae_ld.data.util import natural_sort
import tensorflow as tf


def get_model_filename(model, model_name, param_value):
    m_seed = model.partition("checkpoint")[0].strip("/").split("/")[-1]
    m_short = Path(model).name
    return "{}_{}_seed_{}_{}".format(model_name, param_value, m_seed, m_short)


def get_file_list(model_path, keep_n=0, selection_type="even"):
    m_files = glob(model_path)
    m_files.sort(key=natural_sort)
    if keep_n > 0 and selection_type == "even":
        to_keep = np.linspace(0, len(m_files) - 1, num=keep_n, dtype=np.int32)
        m_files = [m_files[i] for i in to_keep]
    elif keep_n > 0 and selection_type == "first":
        return m_files[:keep_n]
    return m_files


def get_model_epoch(model, dataset_name):
    mstem = Path(model).stem
    i = mstem.find("epoch_")
    # TODO remove hard-coded epochs
    last_epoch = 26
    if dataset_name == "cars3d":
        last_epoch = 1090
    elif dataset_name == "smallnorb":
        last_epoch = 410
    if i == -1:
        epoch = last_epoch
    else:
    # TODO End remove here
        epoch = int(mstem[i:].split("_")[1])
    return epoch


def get_activations(data, model_path):
    model = tf.keras.models.load_model(model_path)
    acts = [data] + model.encoder.predict(data)
    acts += model.decoder.predict(acts[-1])
    # Note that one could get weights using l.get_weights() instead of l.name here
    layer_names = ["input"] + [l.name for l in model.encoder.layers]
    layer_names += [l.name for l in model.decoder.layers]
    return model, acts, layer_names


def get_weights(model_path):
    model = tf.keras.models.load_model(model_path)
    layer_weights = {l.name: l.get_weights() for l in model.encoder.layers}
    layer_weights += {l.name: l.get_weights() for l in model.decoder.layers}
    return layer_weights


def prepare_activations(x):
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    return x