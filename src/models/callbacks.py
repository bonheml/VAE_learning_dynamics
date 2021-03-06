import os
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import sigmoid
from tensorflow.python.distribute import multi_worker_util
import h5py


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
    def __init__(self, *, filepath, data_path, epoch_steps):
        """ Initialisation of svcca callback
        :param filepath: the path used for saving the activations
        :param data_path: the path of the dataset used for retrieving the activation values
        :param epoch_steps: the number of epochs to skip between to save (eg: epoch_steps=5 will save every 5 epochs)
        The dataset is expected to be a npz file containing a data key where the data is stored
        """
        super(SvccaCallback, self).__init__()
        self.filepath = filepath
        data_path = str(Path(data_path).expanduser())
        self.data = np.load(data_path)["data"]
        self.data_size = len(self.data)
        self.layer_names = []
        self.epoch_steps = epoch_steps

    def init_layer_names(self):
        self.layer_names = [layer.name for layer in self.model.encoder.layers[1:]]
        self.layer_names += [layer.name for layer in self.model.decoder.layers[1:]]

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_steps != 0:
            return

        logs = logs if logs else {}
        logs["data_size"] = self.data_size
        file_path = self._get_file_path(logs, epoch)
        logs.pop("data_size")
        if epoch == 0:
            self.init_layer_names()

        encoder_activations = self.model.encoder(self.data, training=False)
        decoder_activations = self.model.decoder(encoder_activations[-1], training=False)
        activations = encoder_activations + decoder_activations
        with h5py.File(file_path, "w") as f:
            for name, act in zip(self.layer_names, activations):
                act = act.numpy()
                f.create_dataset(name, data=act, dtype=act.dtype, compression="gzip")


class ImageGeneratorCallback(FileSaverCallback):
    def __init__(self, *, filepath, nb_samples, data_path, latent_shape):
        """ Initialisation of image generator callback
        :param filepath: the path used for saving the activations
        :param data_path: the path of the dataset used for retrieving the activation values
        The dataset is expected to be a npz file containing a data key where the data is stored
        """
        super(ImageGeneratorCallback, self).__init__()
        self.nb_samples = nb_samples
        self.latent_shape = latent_shape
        self.filepath = filepath
        data_path = str(Path(data_path).expanduser())
        self.data = np.load(data_path)["data"][:nb_samples]

    def save_images(self, generated_images, file_path):
        generated_images = sigmoid(generated_images)
        generated_images *= 255.
        generated_images = generated_images.numpy()
        for i in range(self.nb_samples):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("{}.{}.png".format(file_path, i))

    def generate_images(self, samples, file_path):
        generated_images = self.model.decoder(samples, training=False)
        if type(generated_images) == list:
            generated_images = generated_images[-1]
        self.save_images(generated_images, file_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs else {}
        file_path = self._get_file_path(logs, epoch)
        if epoch == 0:
            self.save_images(self.data, "{}.real_data".format(file_path))
        random_latent_vectors = tf.random.normal(shape=(self.nb_samples, self.latent_shape))
        self.generate_images(random_latent_vectors, "{}.from_random_latents".format(file_path))
        out = self.model.encoder(self.data, training=False)
        _, _, z = out[-3:] if self.model.save_activations else out
        self.generate_images(z, "{}.from_real_data".format(file_path))
