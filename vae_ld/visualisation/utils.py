import matplotlib.pyplot as plt


def save_figure(out_fname, dpi=300, tight=True):
    """ Save a matplotlib figure in an `out_fname` file.

    Parameters
    ----------
    out_fname : str
        Name of the file used to save the figure.
    dpi : int, optional
        Number of dpi, Default 300.
    tight: bool, optional
        If True, use plt.tight_layout() before saving. Default True.

    Returns
    -------
    None
    """
    if tight is True:
        plt.tight_layout()
    plt.savefig(out_fname, dpi=dpi, transparent=True)
    plt.clf()
    plt.cla()
    plt.close()


def add_hatches(ax, legend):
    hatches = ['/', '.', '\\', 'x', '|', '-', '+', '*', 'o', 'O']
    for bars, hatch in zip(ax.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    ax.legend(title=legend)