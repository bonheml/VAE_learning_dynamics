import hydra
import tensorflow as tf
import numpy as np
from hydra.utils import instantiate


def mnist_data():
    # TODO: remove this dataset part once finished
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    return mnist_digits


@hydra.main(config_path="config/config.yaml")
def train(cfg):
    print("{0}\n{2:^80}\n{1}\n{3}{1}\n".format("-" * 80, "=" * 80, "Current model config", cfg.pretty(resolve=True)))
    np.random.seed(cfg.hyperparameters.seed)
    tf.random.set_seed(cfg.hyperparameters.seed)

    latent_shape = cfg.hyperparameters.latent_shape
    input_shape = cfg.dataset.input_shape
    save = cfg.hyperparameters.save_activations

    # TODO: create a proper pipeline for loading data. (one class per dataset that can be instantiated?)
    data = mnist_data()
    optimizer = instantiate(cfg.optimizer)
    rec_loss = instantiate(cfg.reconstruction_loss)
    callbacks = [instantiate(callback) for callback in cfg.callbacks]

    encoder = instantiate(cfg.model.encoder, input_shape=input_shape, output_shape=latent_shape, save_activations=save)
    encoder = encoder.build()
    encoder.summary()

    decoder = instantiate(cfg.model.decoder, input_shape=(latent_shape,), output_shape=input_shape,
                          save_activations=save)
    decoder = decoder.build()
    decoder.summary()

    model_cls = hydra.utils.get_class(cfg.model["class"])
    model = model_cls(encoder=encoder, decoder=decoder, reconstruction_loss_fn=rec_loss, save_activations=save)
    model.compile(optimizer=optimizer, run_eagerly=save)
    model.fit(data, epochs=cfg.hyperparameters.epochs, batch_size=cfg.hyperparameters.batch_size, callbacks=callbacks)


if __name__ == "__main__":
    train()
