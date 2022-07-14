import pathlib

from vae_ld.visualisation import logger
from vae_ld.visualisation.utils import save_figure
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd


def aggregate_ides(input_dir, save_file, overwrite):
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already computed aggregation of {}".format(save_file))
        return
    df = None
    files = glob("{}/*.tsv".format(input_dir))
    logger.debug("Files to process are {}".format(files))
    for f in files:
        df2 = pd.read_csv(f, sep="\t", index_col=0)
        if df is not None:
            df = pd.concat([df, df2], ignore_index=True)
        else:
            df = df2
    logger.debug("Aggregated dataframe:\n{}".format(df))
    df.to_csv(save_file, sep="\t", index=False)


def plot_latents_ides(input_file, save_file, overwrite, xy_annot=None, xy_text=None):
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already computed latent ides of {}".format(save_file))
        return
    df = pd.read_csv(input_file, sep="\t")
    df2 = df[df.layer.isin(["encoder/z_mean", "encoder/z_log_var", "sampling"]) &
             df.estimator.isin(["MLE_10", "MLE_20", "TwoNN"])]
    df2 = df2.rename(columns={"latent_dim": "Number of latent dimensions", "layer": "Representation"})
    df2 = df2.replace("encoder/z_mean", "Mean")
    df2 = df2.replace("encoder/z_log_var", "Variance")
    df2 = df2.replace("sampling", "Sampled")
    ax = sns.lineplot(data=df2, x="Number of latent dimensions", y="IDE", hue="Representation", style="Representation",
                      linewidth=4)
    if xy_text is not None and xy_annot is not None:
        ax.annotate("Optimal", xy=xy_annot, xycoords='data', xytext=xy_text, textcoords='data', fontsize=15,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black', lw=3.5, ls='--'))
    save_figure(save_file)


def plot_data_ides(input_dir, save_file, overwrite):
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
    df2 = df2.rename(columns={"estimator": "Estimator"})
    _ = sns.barplot(x="Dataset", y="IDE", hue="Estimator", data=df2)
    save_figure(save_file)


def plot_layers_ides(input_file, save_file, overwrite):
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already computed layers ides of {}".format(save_file))
        return
    df = pd.read_csv(input_file, sep="\t")
    df = df[df.estimator.isin(["MLE_10", "MLE_20", "TwoNN"])]
    df = df.replace("encoder/z_mean", "Mean")
    df = df.replace("encoder/z_log_var", "Variance")
    df = df.replace("sampling", "Sampled")

    sns.pointplot(x="layer", y="IDE", data=df)
    _ = plt.xticks(
        rotation=45,
        horizontalalignment='right',
    )
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    save_figure(save_file)
