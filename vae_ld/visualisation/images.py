import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from vae_ld.visualisation.utils import save_figure


def plot_and_save(imgs, fname, samples=None):
    r = int(np.floor(np.sqrt(len(imgs))))
    c = r if samples is None else r * 2
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111, nrows_ncols=(r, c), axes_pad=0, share_all=True)

    to_process = []
    if samples is None:
        to_process = imgs
    else:
        for t in zip(imgs, samples):
            to_process += t

    for ax, im in zip(grid, to_process):
        ax.imshow(im)

    fig.subplots_adjust(wspace=0, hspace=0)
    save_figure(fname)