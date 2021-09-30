import os
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import sigmoid
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
                return self.filepath.format(epoch=epoch, **logs)
            except KeyError as e:
                raise KeyError("Failed to format this callback filepath: \"{}\". Reason: {}".format(self.filepath, e))
        else:
            self._temp_file_dir = tempfile.mkdtemp()
            extension = os.path.splitext(self.filepath)[1]
            return os.path.join(self._temp_file_dir, "temp" + extension)


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
        _, _, z = out[-3:]
        self.generate_images(z, "{}.from_real_data".format(file_path))
