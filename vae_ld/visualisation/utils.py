import matplotlib.pyplot as plt


def save_figure(out_fname, dpi=300, tight=True):
    if tight is True:
        plt.tight_layout()
    plt.savefig(out_fname, dpi=dpi)
    plt.clf()
    plt.cla()
    plt.close()
