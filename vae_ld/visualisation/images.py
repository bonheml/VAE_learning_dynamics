import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from vae_ld.visualisation.utils import save_figure


def plot_and_save(imgs, fname, greyscale, samples=None):
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
    save_figure(fname, tight=False)


def plot_conv_layers(samples, fname):
    if len(samples.shape) == 3:
        return plot_conv_layer(samples, "example_0_{}".format(fname))

    for i, outputs in enumerate(samples):
        fname = "example_{}_{}".format(i, fname)
        plot_conv_layer(outputs, fname)


def plot_conv_layer(outputs, fname):
    cols = int(np.ceil(np.sqrt(outputs.shape[-1])))
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111, nrows_ncols=(cols, cols), axes_pad=0)

    for ax, out in zip(grid, outputs):
        ax.set_axis_off()
        ax.matshow(out)

    fig.subplots_adjust(wspace=0, hspace=0)
    save_figure(fname, tight=False)
