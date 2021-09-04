import h5py


def load_layers_from_file(file_path):
    """ Load layers activation from a given file.

    :param file_path: the path of the file to load
    :type file_path: str
    :return: A dictionary of the layer activations
    :rtype: dict
    """
    with h5py.File(file_path, "r") as f:
        layers = {"{}/{}".format(grp_name, ds_name): ds[()] for (grp_name, grp) in f.items()
                  for (ds_name, ds) in grp.items()}
    return layers

