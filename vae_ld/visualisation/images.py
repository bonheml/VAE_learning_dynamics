import numpy as np
import matplotlib.pyplot as plt
from vae_ld.visualisation.utils import save_figure


def plot_and_save(imgs, fname):
    n = int(np.sqrt(imgs[0].shape[0]))
    n_sources = len(imgs)
    fig = plt.figure(figsize=(n * n_sources, n))

    for i in range(0, imgs[0].shape[0]):
        for j in range(n_sources):
            plt.subplot(n * n_sources, n, (n_sources * i) + j + 1)
            plt.imshow(imgs[j][i, :, :, 0], cmap='gray')
            plt.axis('off')

    fig.subplots_adjust(wspace=0, hspace=0)
    save_figure(fname)
