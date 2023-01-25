import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from vae_ld.visualisation.utils import save_figure
import tensorflow as tf


def plot_and_save(imgs, fname, greyscale, samples=None, r=None, c=None):
    if r and c:
        plot_traversal(imgs, r, c, greyscale)
    else:
        plot_images(imgs, greyscale, samples=samples)
    save_figure(fname, tight=False)


def plot_images(imgs, greyscale, samples=None, show=False):
    r = int(np.floor(np.sqrt(len(imgs))))
    c = r if samples is None else r * 2
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111, nrows_ncols=(r, c), axes_pad=0)

    to_process = []
    if samples is None:
        to_process = imgs
    else:
        for t in zip(samples, imgs):
            to_process += t

    for ax, im in zip(grid, to_process):
        ax.set_axis_off()
        if greyscale is True:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)

    fig.subplots_adjust(wspace=0, hspace=0)
    if show is True:
        plt.show()


def plot_traversal(imgs, r, c, greyscale, show=False):
    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111, nrows_ncols=(r, c), axes_pad=0, direction="column")

    for i, (ax, im) in enumerate(zip(grid, imgs)):
        ax.set_axis_off()
        if i % r == 0:
            ax.set_title("z{}".format(i // r), fontdict={'fontsize': 25})
        if greyscale is True:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)

    fig.subplots_adjust(wspace=0, hspace=0)
    if show is True:
        plt.show()


def plot_conv_layers(outputs, base_fname):

    for i, sample in enumerate(outputs):
        cols = int(np.ceil(np.sqrt(sample.shape[-1])))
        fname = "example_{}_{}".format(i, base_fname)
        fig = plt.figure(figsize=(10., 10.))
        grid = ImageGrid(fig, 111, nrows_ncols=(cols, cols))

        for j, ax in enumerate(grid):
            ax.set_axis_off()
            if j < sample.shape[-1]:
                ax.matshow(sample[:, :, j])

        fig.subplots_adjust(wspace=0, hspace=0)
        save_figure(fname, tight=False)


def latent_traversal(model_path, fname, samples, greyscale, n_changes=10, val_range=(-1, 1)):
    m = tf.keras.models.load_model(model_path)
    z_base = m.encoder(samples)[-1]
    r, c = n_changes, z_base.shape[1]
    vals = np.linspace(*val_range, r)
    for j, z in enumerate(z_base):
        imgs = np.empty([r * c, 64, 64, 1]) if greyscale else np.empty([r * c, 64, 64, 3])
        for i in range(c):
            z_iter = np.tile(z, [r, 1])
            z_iter[:, i] = vals
            imgs[r * i:(r * i) + r] = tf.sigmoid(m.decoder(z_iter, training=False)[-1])
        plot_and_save(imgs, "{}_example_{}.pdf".format(fname, j), greyscale, r=r, c=c)
