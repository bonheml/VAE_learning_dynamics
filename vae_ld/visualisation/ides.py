import pathlib

import numpy as np

from vae_ld.visualisation import logger
from vae_ld.visualisation.utils import save_figure, add_hatches
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

sns.set(rc={'figure.figsize': (10, 10)}, font_scale=3)
sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})


def aggregate_ides(input_dir, save_file, overwrite):
    """Aggregate IDEs results in `input_dir` and save the results to `save_file` in TSV format.

    Parameters
    ----------
    input_dir : str
        Absolute path to the directory where the similarity scores are stored
    save_file : str
        Name of the file used to save the aggregated results
    overwrite : bool
        If True, overwrite `save_file` content if `save_file` already exists.

    Returns
    -------
    None
    """
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already computed aggregation of {}".format(save_file))
        return
    df = None
    files = glob(input_dir)
    logger.debug("Files to process are {}".format(files))
    for f in files:
        df2 = pd.read_csv(f, sep="\t", index_col=0)
        if df is not None:
            df = pd.concat([df, df2], ignore_index=True)
        else:
            df = df2
    logger.debug("Aggregated dataframe:\n{}".format(df))
    df.to_csv(save_file, sep="\t", index=False)


def plot_latents_ides(input_file, save_file, overwrite, xy_annot=None, xy_text=None, text=None):
    """ Plot a line plot of the IDE of mean, variance and sampled representations

    Parameters
    ----------
    input_file : str
        Name of the file containing the aggregated results
    save_file : str
        Name of the file used to save the figure.
    overwrite : bool
        If True, overwrite any existing file, else skip them.
    xy_annot : tuple, optional
        Where to start the arrow. If None, no annotation is performed
    xy_text : tuple, optional
        Where to end the arrow. If None, no annotation is performed
    text : str, optional
        The text of the annotation. If None, no annotation is performed

    Returns
    -------
    None
    """
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already computed latent ides of {}".format(save_file))
        return
    df = pd.read_csv(input_file, sep="\t")
    df2 = df[df.layer.isin(["encoder/z_mean", "encoder/z_log_var", "sampling"]) &
             df.estimator.isin(["MLE_10", "MLE_20", "TwoNN"])]
    df2 = df2.rename(columns={"latent_dim": "Number of latent dimensions"})
    df2 = df2.replace("encoder/z_mean", "Mean")
    df2 = df2.replace("encoder/z_log_var", "Variance")
    df2 = df2.replace("sampling", "Sampled")
    ax = sns.lineplot(data=df2, x="Number of latent dimensions", y="IDE", hue="layer", style="layer",
                      linewidth=10, ci="sd")
    ax.legend(title="Representation")
    for line in plt.legend().get_lines():
        line.set_linewidth(8)
    if xy_text is not None and xy_annot is not None:
        text = "" if not text else text
        ax.annotate(text, xy=xy_annot, xycoords='data', xytext=xy_text, textcoords='data', fontsize=30,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black', lw=6, ls='--'))
    save_figure(save_file)


def plot_data_ides(input_dir, save_file, overwrite):
    """ Do a bar plot of the data IDEs

    Parameters
    ----------
    input_dir: str
        Name of the directory containing the results
    save_file : str
        Name of the file used to save the figure.
    overwrite : bool
        If True, overwrite any existing file, else skip them.

    Returns
    -------
    None
    """
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already data ides of {}".format(save_file))
        return
    files = glob("{}/*.tsv".format(input_dir))
    df = None
    for file in files:
        ds_name = file.split("/")[-1].split("_")[1].capitalize()
        df2 = pd.read_csv(file, sep="\t")
        df2["Dataset"] = ds_name
        df = df2 if df is None else pd.concat([df, df2], ignore_index=True)
    df2 = df[df.layer == "input"]
    df2 = df2.replace("Dsprites", "dSprites")
    df2 = df2.replace(["MLE_3", "MLE_5", "MLE_10", "MLE_20"], ["MLE(k=3)", "MLE(k=5)", "MLE(k=10)", "MLE(k=20)"])
    ax = sns.barplot(x="Dataset", y="IDE", hue="estimator", data=df2, order=["Symsol", "dSprites", "Celeba"],
                     ci="sd", palette="winter")
    add_hatches(ax, "Estimator")
    save_figure(save_file)


def plot_layers_ides(input_file, save_file, overwrite, hue="Number of latent dimensions"):
    """ Plot a line plot of the IDE of every VAE layer

    Parameters
    ----------
    input_file : str
        Name of the file containing the aggregated results
    save_file : str
        Name of the file used to save the figure.
    overwrite : bool
        If True, overwrite any existing file, else skip them.
    hue : str, optional
        Column of the dataframe on which the hue should be applied.
        default is Number of latent dimensions

    Returns
    -------
    None
    """
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already computed layers ides of {}".format(save_file))
        return
    sns.set(rc={'figure.figsize': (15, 10), 'lines.linewidth': 2}, font_scale=3)
    sns.set_style("whitegrid", {'axes.grid': False})
    df = pd.read_csv(input_file, sep="\t")
    df = df[df.estimator.isin(["MLE_10", "MLE_20", "TwoNN"])]
    df = df[df.layer != "sampling_1"]
    df = df.replace("encoder/z_mean", "mean")
    df = df.replace("encoder/z_log_var", "variance")
    df = df.replace("sampling", "sampled")
    df = df[df.layer != "decoder/reshape"]
    df = df.rename(columns={"latent_dim": "Number of latent dimensions", "layer": "Layer", "model_epoch": "Epoch"})
    markers = ["D", "v", "o", "^", "s", "<", ">", "p", "*", "X", ".", "8", "d", "H"]
    ax = sns.pointplot(x="Layer", y="IDE", hue=hue, markers=markers, data=df, ci="sd")
    _ = plt.xticks(
        rotation=90,
        horizontalalignment='center',
    )
    ax.set_ylim(0, np.ceil(df.IDE.max()))
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labelspacing=0.3)
    save_figure(save_file)
