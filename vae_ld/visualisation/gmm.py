import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
from vae_ld.visualisation.utils import save_figure
from vae_ld.visualisation import logger

sns.set(rc={'figure.figsize': (10, 10)}, font_scale=3)
sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})


def plot_hist(X, save_file, variable_name=r"$\mathbf{\sigma}$", xlims=None, bins=13):
    fig, ax = plt.subplots()
    ax.hist(X, density=True, histtype="stepfilled", bins=bins)
    ax.set_xlabel(variable_name)
    ax.set_ylabel('Density')
    if xlims is not None:
        plt.xlim(*xlims)
    save_figure(save_file)


def plot_traversal(imgs, r, c, greyscale, fname, show=False):
    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111, nrows_ncols=(r, c), axes_pad=0, direction="row")

    for i, (ax, im) in enumerate(zip(grid, imgs)):
        ax.set_axis_off()
        if greyscale is True:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)

    fig.subplots_adjust(wspace=0, hspace=0)
    if show is True:
        plt.show()
    save_figure(fname, tight=False)


def latent_traversal(decoder, fname, z_base, greyscale, idx, n_changes=5, val_range=3):
    r, c = z_base.shape[0], n_changes
    imgs = np.empty([r, c, 64, 64, 1]) if greyscale else np.empty([r, c, 64, 64, 3])

    for i, z in enumerate(z_base):
        logger.info("z[{}] = {}".format(idx, z[idx]))
        vals = np.linspace(z[idx] - val_range, z[idx] + val_range, num=n_changes)
        z_iter = np.tile(z.reshape(1, -1), [n_changes, 1])
        z_iter[:, idx] = vals
        imgs[i] = tf.sigmoid(decoder(z_iter, training=False)[-1])
    plot_traversal(imgs.reshape((r * c, *imgs.shape[2:])), r, c, greyscale, fname)

