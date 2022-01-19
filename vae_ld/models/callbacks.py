import tensorflow as tf
from tensorflow import sigmoid
from vae_ld.visualisation.images import plot_and_save


class ImageGeneratorCallback(tf.keras.callbacks.Callback):
    def __init__(self, *, filepath, nb_samples, data, latent_shape, save_freq):
        """ Initialisation of image generator callback
        :param nb_samples: The number of examples to generate
        :param data: the dataset used for sampling examples
        """
        super(ImageGeneratorCallback, self).__init__()
        self.nb_samples = nb_samples
        self.latent_shape = latent_shape
        self.data = data[:nb_samples]
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            random_latent_vectors = tf.random.normal(shape=(self.nb_samples, self.latent_shape))
            generated_images = sigmoid(self.model.decoder(random_latent_vectors, training=False)[-1]) * 255.
            plot_and_save(generated_images, "{}/epoch_{}_from_random_latents.pdf".format(self.filepath, epoch))

            z = self.model.encoder(self.data, training=False)[-1]
            generated_images = sigmoid(self.model.decoder(z, training=False)[-1]) * 255.
            plot_and_save(generated_images, "{}/epoch_{}_from_real_data.pdf".format(self.filepath, epoch), self.data)

