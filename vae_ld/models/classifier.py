import tensorflow as tf
from tensorflow.keras import layers


class Classifier(tf.keras.Model):
    """ Classifier using similar architecture to VAE encoders
    """

    def __init__(self, *, clf, input_shape, **kwargs):
        """ Model initialisation

        :param clf: the classifier architecture to use (must be initialised beforehand).
        :param input_shape: the shape of the input
        """
        super(Classifier, self).__init__(**kwargs)
        self.clf = clf
        self.clf.build((None, *input_shape))
        self.clf.summary()
        # This is needed to save the model properly
        self.built = True
        self.classification_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.classification_loss_tracker = tf.keras.metrics.Mean(name="classification_loss")

    @property
    def metrics(self):
        return [self.classification_loss_tracker]

    def call(self, inputs):
        return self.clf(inputs)[-1]

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.clf(x, training=True)[-1]
            loss = tf.reduce_mean(self.classification_loss_fn(y, y_pred))

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.classification_loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.clf(x, training=False)[-1]
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


class ConvolutionalClassifier(tf.keras.Model):
    """ Convolutional classifier based on the encoder architecture initially used in beta-VAE [1].
    Based on Locatello et al. [2] implementation (https://github.com/google-research/disentanglement_lib)

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
    """
    def __init__(self, input_shape, output_shape):
        # If output shape is a list of labels, only use the first one for the classification.
        try:
            output_shape = output_shape[0]
        except TypeError:
            pass

        super(ConvolutionalClassifier, self).__init__()
        self.e1 = layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=4, strides=2, activation="relu",
                                padding="same", name="classifier/1")
        self.e2 = layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu", padding="same",
                                name="classifier/2")
        self.e3 = layers.Conv2D(filters=64, kernel_size=2, strides=2, activation="relu", padding="same",
                                name="classifier/3")
        self.e4 = layers.Conv2D(filters=64, kernel_size=2, strides=2, activation="relu", padding="same",
                                name="classifier/4")
        self.e5 = layers.Flatten(name="classifier/5")
        self.e6 = layers.Dense(256, activation="relu", name="classifier/6")
        self.e7 = layers.Dense(output_shape, name="classifier/output", activation="softmax")

    def call(self, inputs):
        x1 = self.e1(inputs)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        out = self.e7(x6)
        return x1, x2, x3, x4, x5, x6, out


class FullyConnectedClassifier(tf.keras.Model):
    """ Fully connected classifier based on the encoder architecture initially used in beta-VAE [1].
    Based on Locatello et al. [2] implementation (https://github.com/google-research/disentanglement_lib)

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
    """
    def __init__(self, input_shape, output_shape):
        super(FullyConnectedClassifier, self).__init__()
        self.e1 = layers.Flatten(name="classifier/1", input_shape=input_shape)
        self.e2 = layers.Dense(1200, activation="relu", name="classifier/2")
        self.e3 = layers.Dense(1200, activation="relu", name="classifier/3")
        self.e7 = layers.Dense(output_shape, name="classifier/output", activation="softmax")

    def call(self, inputs):
        x1 = self.e1(inputs)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        out= self.z_mean(x3)
        return x1, x2, x3, out