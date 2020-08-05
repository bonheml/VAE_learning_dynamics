import os
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import multi_worker_util


class FileSaverCallback(tf.keras.callbacks.Callback):
    """ Generic class implementing get_file_path to retrieve create a file name using log content
    """
    def _get_file_path(self, logs, epoch):
        """ Returns the file path for checkpoint. Similarly to tf.keras.callbacks.ModelCheckpoint
        """
        # noinspection PyProtectedMember
        if not self.model._in_multi_worker_mode() or multi_worker_util.should_save_checkpoint():
            try:
                return self.filepath.format(epoch=epoch+1, **logs)
            except KeyError as e:
                raise KeyError("Failed to format this callback filepath: \"{}\". Reason: {}".format(self.filepath, e))
        else:
            self._temp_file_dir = tempfile.mkdtemp()
            extension = os.path.splitext(self.filepath)[1]
            return os.path.join(self._temp_file_dir, "temp" + extension)


class SvccaCallback(FileSaverCallback):
    def __init__(self, *, filepath, data_path):
        """ Initialisation of svcca callback
        :param filepath: the path used for saving the activations
        :param data_path: the path of the dataset used for retrieving the activation values
        The dataset is expected to be a npz file containing a data key where the data is stored
        """
        super(SvccaCallback, self).__init__()
        self.filepath = filepath
        data_path = str(Path(data_path).expanduser())
        self.data = np.load(data_path)["data"]
        self.data_size = len(self.data)
        self.layer_names = []

    def init_layer_names(self):
        self.layer_names = [layer.name for layer in self.model.encoder.layers[1:]]
        self.layer_names += [layer.name for layer in self.model.decoder.layers[1:]]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs else {}
        logs["data_size"] = self.data_size
        file_path = self._get_file_path(logs, epoch)
        logs.pop("data_size")
        if epoch == 0:
            self.init_layer_names()

        layers_activations = {}
        encoder_layers = self.model.encoder(self.data)
        decoder_layers = self.model.decoder(encoder_layers[-1])
        activations = encoder_layers + decoder_layers
        for i, k in enumerate(self.layer_names):
            layers_activations[k] = activations[i].numpy()
        np.savez_compressed(file_path, **layers_activations)


class ImageGeneratorCallback(FileSaverCallback):
    def __init__(self, *, filepath, nb_samples, latent_shape):
        super(ImageGeneratorCallback, self).__init__()
        self.nb_samples = nb_samples
        self.latent_shape = latent_shape
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs else {}
        file_path = self._get_file_path(logs, epoch)
        random_latent_vectors = tf.random.normal(shape=(self.nb_samples, self.latent_shape))
        generated_images = self.model.decoder(random_latent_vectors)
        if type(generated_images) == list:
            generated_images = generated_images[-1]
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.nb_samples):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("{}.{}.png".format(file_path, i))
