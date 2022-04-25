import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from vae_ld.models import logger


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
        self.classification_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.classification_loss_tracker = [tf.keras.metrics.Mean(name="Categorical cross-entropy factor {}".format(i))
                                            for i in range(clf.n_classes)]
        self.classification_accuracy_fn = tf.keras.metrics.CategoricalAccuracy()
        self.classification_accuracy_tracker = [tf.keras.metrics.Mean(name="Categorical accuracy factor {}".format(i))
                                                for i in range(clf.n_classes)]
        self.model_loss_tracker = tf.keras.metrics.Mean(name="model_loss")

    @property
    def metrics(self):
        return [*self.classification_loss_tracker, *self.classification_accuracy_tracker, self.model_loss_tracker]

    def call(self, inputs):
        return self.clf(inputs)[-1]

    def train_step(self, data):
        x, y = data
        logger.debug("Receive batch of {} labels".format(y.shape))
        logger.debug("Labels are {}".format(y))

        with tf.GradientTape() as tape:
            y_pred = self.clf(x, training=True)[-1]
            losses = []
            logger.debug("Receive batch of ({},{}) predictions".format(len(y_pred), y_pred[0].shape[0]))
            logger.debug("Predictions are {}".format(y_pred))
            for i in y_pred:
                acc = self.classification_accuracy[i](y[i], y_pred[i])
                self.classification_accuracy_tracker[i].update_state(acc)
                loss = self.classification_loss_fn(y[i], y_pred[i])
                losses.append(loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(losses, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for i, loss in enumerate(losses):
            self.classification_loss_tracker[i].update_state(loss)
        self.model_loss_tracker.update_state(np.sum(losses))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.clf(x, training=False)[-1]

        losses = 0
        for i in y_pred:
            acc = self.classification_accuracy[i](y[i], y_pred[i])
            self.classification_accuracy_tracker[i].update_state(acc)
            loss = self.classification_loss_fn(y[i], y_pred[i])
            self.classification_loss_tracker[i].update_state(loss)
            losses += loss
        self.model_loss_tracker.update_state(losses)

        return {m.name: m.result() for m in self.metrics}


class ConvolutionalClassifier(tf.keras.Model):
    """ Convolutional classifier based on the encoder architecture initially used in beta-VAE [1].
    Based on Locatello et al. [2] implementation (https://github.com/google-research/disentanglement_lib)
    Here we do multi-ouput classification instead of generating sampled representations, thus the final layers are softmax layers.

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
    """
    def __init__(self, input_shape, output_shape):
        super(ConvolutionalClassifier, self).__init__()
        self.n_classes = len(output_shape)
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

        self.outputs = []
        for i, factor_shape in enumerate(output_shape):
            # Note that here we keep a softmax activation because all our data have a finite number of values for
            # each factor. If we introduce a new dataset with continuous values, this will need to be updated
            # to allow other activations functions (e.g., sigmoid, tanh)
            self.outputs.append(layers.Dense(factor_shape, name="classifier/output_{}".format(i), activation="softmax"))

    def call(self, inputs):
        x1 = self.e1(inputs)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        out = [output(x6) for output in self.outputs]
        return x1, x2, x3, x4, x5, x6, out


class FullyConnectedClassifier(tf.keras.Model):
    """ Fully connected classifier based on the encoder architecture initially used in beta-VAE [1].
    Based on Locatello et al. [2] implementation (https://github.com/google-research/disentanglement_lib)
    Here we do multi-ouput classification instead of generating sampled representations, thus the final layers are softmax layers.

    [1] Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
    In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France.
    [2] Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
    Representations. In K. Chaudhuri and R. Salakhutdinov, eds., Proceedings of the 36th International Conference
    on Machine Learning, Proceedings of Machine Learning Research, vol. 97, Long Beach, California, USA: PMLR,
    pp. 4114–4124.
    """
    def __init__(self, input_shape, output_shape):
        super(FullyConnectedClassifier, self).__init__()
        self.n_classes = len(output_shape)
        self.e1 = layers.Flatten(name="classifier/1", input_shape=input_shape)
        self.e2 = layers.Dense(1200, activation="relu", name="classifier/2")
        self.e3 = layers.Dense(1200, activation="relu", name="classifier/3")
        self.outputs = []
        for i, factor_shape in enumerate(output_shape):
            # Note that here we keep a softmax activation because all our data have a finite number of values for
            # each factor. If we introduce a new dataset with continuous values, this will need to be updated
            # to allow other activations functions (e.g., sigmoid, tanh)
            self.outputs.append(layers.Dense(factor_shape, name="classifier/output_{}".format(i), activation="softmax"))

    def call(self, inputs):
        x1 = self.e1(inputs)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        out = [output(x3) for output in self.outputs]
        return x1, x2, x3, out
