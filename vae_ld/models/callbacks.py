import tensorflow as tf
from tensorflow import sigmoid


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

    def save_images(self, generated_images, file_path):
        generated_images = sigmoid(generated_images)
        generated_images *= 255.
        generated_images = generated_images.numpy()
        for i in range(self.nb_samples):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("{}.{}.png".format(file_path, i))

    def generate_images(self, samples, file_path):
        generated_images = self.model.decoder(samples, training=False)[-1]
        self.save_images(generated_images, file_path)

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.save_images(self.data, "{}/data_example".format(self.filepath, epoch))
        if epoch % self.save_freq == 0:
            random_latent_vectors = tf.random.normal(shape=(self.nb_samples, self.latent_shape))
            self.generate_images(random_latent_vectors, "{}/epoch_{}_from_random_latents".format(self.filepath, epoch))
            z = self.model.encoder(self.data, training=False)[-1]
            self.generate_images(z, "{}/epoch_{}_from_real_data".format(self.filepath, epoch))

