import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import multi_worker_util


class FileSaverCallback(tf.keras.callbacks.Callback):
    """ Generic class implementing get_file_path to retrieve create a file name using log content
    """
    def _get_file_path(self, epoch, logs):
        """ Returns the file path for checkpoint. Similarly to tf.keras.callbacks.ModelCheckpoint
        """
        # noinspection PyProtectedMember
        if not self.model._in_multi_worker_mode() or multi_worker_util.should_save_checkpoint():
            try:
                return self.filepath.format(epoch=epoch + 1, **logs)
            except KeyError as e:
                raise KeyError("Failed to format this callback filepath: \"{}\". Reason: {}".format(self.filepath, e))
        else:
            self._temp_file_dir = tempfile.mkdtemp()
            extension = os.path.splitext(self.filepath)[1]
            return os.path.join(self._temp_file_dir, "temp" + extension)


class SvccaCallback(FileSaverCallback):
    def __init__(self, *, filepath):
        super(SvccaCallback, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs else {}
        file_path = self._get_file_path(epoch, logs)
        activations = [tf.concat(a, 0).numpy() for a in self.model.layers_activations.values()]
        to_save = dict(zip(self.model.layers_activations.keys(), activations))
        np.savez_compressed(file_path, **to_save)
        self.model.init_layers_activations()
        print('\nSaved layers activations at epoch: {}'.format(epoch))


class ImageGeneratorCallback(FileSaverCallback):
    def __init__(self, *, filepath, nb_samples, latent_shape):
        super(ImageGeneratorCallback, self).__init__()
        self.nb_samples = nb_samples
        self.latent_shape = latent_shape
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs else {}
        file_path = self._get_file_path(epoch, logs)
        random_latent_vectors = tf.random.normal(shape=(self.nb_samples, self.latent_shape))
        generated_images = self.model.decoder(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.nb_samples):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("{}.{}".format(file_path, i))
