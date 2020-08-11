import h5py


def load_layers_from_file(file_path, layers_name="all"):
    """ Load layers activation from a given file.

    :param file_path: the path of the file to load
    :type file_path: str
    :param layers_name: the names of the layer to load. Can be a list of layer names or 'all' to load all the layers.
    Default value is 'all'.
    :type layers_name: Union[str, list]
    :return: A dictionary of the layer activations
    :rtype: dict
    """
    layers = {}
    with h5py.File(file_path, "r") as f:
        if layers_name == "all":
            layers_name = f.keys()
        for layer_name in layers_name:
            layers[layer_name] = f[layer_name][:]

    return layers


def load_layers_from_files(files_path, layers_name="all"):
    """ Load layers activation from multiple files.

    :param files_path: the paths of the files to load
    :type files_path: list
    :param layers_name: the names of the layer to load. Can be a list of layer names or 'all' to load all the layers.
    Default value is 'all'.
    :type layers_name: Union[str, list]
    :return: A dictionary of the layer activations
    :rtype: dict
    """
    layers = {}
    for file in files_path:
        layers[file] = load_layers_from_file(file, layers_name)
    return layers

