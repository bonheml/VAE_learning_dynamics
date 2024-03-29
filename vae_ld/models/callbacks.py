import tensorflow as tf
from tensorflow import sigmoid
import numpy as np
from vae_ld.models import logger
from vae_ld.visualisation.images import plot_and_save


class ImageGeneratorCallback(tf.keras.callbacks.Callback):
    """ Callback saving image grid generated by the training VAEs

    Parameters
    ----------
    filepath : str
        path where the images will be saved
    nb_samples : int
        Number of images in to put in the grid
    data : tf.Tensor
        The data to use as input
    latent_shape : int
        The number of latent dimensions
    save_freq : int
        Save images every save_freq
    greyscale : bool
        True if images are greyscale, else False
    """
    def __init__(self, *args, filepath, nb_samples, data, latent_shape, save_freq, greyscale):
        super(ImageGeneratorCallback, self).__init__(*args)
        self.nb_samples = nb_samples
        self.latent_shape = latent_shape
        self.data = list(data)
        self.multi_input = len(data) == 2
        self.filepath = filepath
        self.save_freq = save_freq
        self.greyscale = greyscale

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            random_latent_vectors = tf.random.normal(shape=(self.nb_samples, self.latent_shape))
            generated_images = sigmoid(self.model.decoder(random_latent_vectors)[-1])
            plot_and_save(generated_images, "{}/epoch_{}_from_random_latents.pdf".format(self.filepath, epoch),
                          self.greyscale)
            generated_images = sigmoid(self.model(self.data, training=False))
            logger.debug("Generated reconstruction of data with shape {}".format(generated_images.shape))
            x = self.data[0] if self.multi_input else self.data
            plot_and_save(generated_images, "{}/epoch_{}_from_real_data.pdf".format(self.filepath, epoch),
                          self.greyscale, x)
