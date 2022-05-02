import pathlib
import numpy as np
import seaborn as sns
from glob import glob
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from vae_ld.visualisation import logger
from vae_ld.visualisation.utils import save_figure
import matplotlib.pyplot as plt

sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})


def check_exists(save_file):
    save_file = save_file.replace("/", "_")
    return save_file, pathlib.Path(save_file).exists()


def similarity_heatmap(metric_name, input_file, overwrite):
    """ Compute heatmaps of similarity scores obtained with `metric_name` for each model.

    Parameters
    ----------
    metric_name : str
        Name of the similarity metric used
    input_file : str
        Name of the file containing the aggregated results
    overwrite : bool
        If True, overwrite any existing file, else skip them.

    Returns
    -------
    None
    """
    df = pd.read_csv(input_file, sep="\t", header=0)
    # Sampling layer is named differently on linear architectures
    df.replace("encoder/z", "sampling", inplace=True)
    grouped_df = df.groupby(["m1_name", "p1_value", "m1_seed", "m1_epoch", "m2_name", "p2_value", "m2_seed",
                             "m2_epoch"])
    group_names = grouped_df.groups.keys()
    # Handle different number of layers for different encoder/decoder architectures.
    encoder_layers1 = ["encoder/{}".format(i) for i in range(1, 7) if "encoder/{}".format(i) in df["m1"].values]
    decoder_layers1 = ["decoder/{}".format(i) for i in range(1, 7) if "decoder/{}".format(i) in df["m1"].values]
    col_order1 = (["input"] + encoder_layers1 + ["encoder/z_mean", "encoder/z_log_var", "sampling"] + decoder_layers1)

    encoder_layers2 = ["encoder/{}".format(i) for i in range(1, 7) if "encoder/{}".format(i) in df["m2"].values]
    decoder_layers2 = ["decoder/{}".format(i) for i in range(1, 7) if "decoder/{}".format(i) in df["m2"].values]
    col_order2 = (["input"] + encoder_layers2 + ["encoder/z_mean", "encoder/z_log_var", "sampling"] + decoder_layers2)

    for group_name in group_names:
        cfg = "{}, param={}, seed={}, epoch={} and {}, param={}, seed={}, epoch={}".format(*group_name)
        save_path = "{}_{}_seed_{}_epoch_{}_{}_{}_seed_{}_epoch_{}.pdf".format(*group_name)

        if overwrite is False and pathlib.Path(save_path).exists():
            logger.info("Skipping already plotted heatmap of {}".format(cfg))
            continue

        group = grouped_df.get_group(group_name)
        logger.info("Plotting heatmap of {}".format(cfg))
        ax = sns.heatmap(group.pivot("m1", "m2", metric_name).reindex(index=col_order1, columns=col_order2), vmin=0, vmax=1)
        ax.set(ylabel="{}, {}={}, seed={}, epoch={}".format(group_name[0], group["p1_name"].values[0], *group_name[1:4]),
               xlabel="{}, {}={}, seed={}, epoch={}".format(group_name[4], group["p2_name"].values[0], *group_name[5:]))
        save_figure(save_path)


def avg_similarity_layer_pair(metric_name, input_file, m1_layer, m2_layer, save_file, overwrite):
    """ Returns a lines plot of the average similarity values between two layers
    over different epochs with different regularisation strength (one line per regularisation weight).

    Parameters
    ----------
    metric_name : str
        Name of the similarity metric used
    input_file : str
        Name of the file containing the aggregated results
    m1_layer : str
        Name of the first layer
    m2_layer : str
        Name of the second layer
    save_file : str
        Name of the file used to save the figure.
    overwrite : bool
        If True, overwrite any existing file, else skip them.

    Returns
    -------
    None
    """
    save_file, exists = check_exists(save_file)
    if exists and overwrite is False:
        logger.info("Skipping already computed layer pair of {}".format(save_file))
        return

    df = pd.read_csv(input_file, sep="\t", header=0)
    # Keep only similarity between identical runs, epochs and models
    df = df[(df["m1_name"] == df["m2_name"]) & (df["m1_seed"] == df["m2_seed"]) & (df["p1_value"] == df["p2_value"])
            & (df["m1_epoch"] == df["m2_epoch"])]
    param = df["p1_name"].values[0]
    # Keep only similarity between m1 and m2 layers
    df = df[(df["m1"] == m1_layer) & (df["m2"] == m2_layer)]
    df.rename(columns={"p1_value": param, "m1_epoch": "epoch"}, inplace=True)
    ax = sns.lineplot(data=df, x="epoch", y=metric_name, hue=param, style=param)
    ax.set(ylim=(0, 1))
    save_figure(save_file)


def avg_similarity_layer_list(metric_name, input_file, regularisation, layer, target, save_file, overwrite):
    """ Returns a lines plot of the average similarity values between `layer` and each `target` layers
    over different epochs.

    Parameters
    ----------
    metric_name : str
        Name of the similarity metric used
    input_file : str
        Name of the file containing the aggregated results
    regularisation : int or float
        The value of the model's regularisation weight
    layer : str
        Name of the layer to use
    target : str
        Can be "encoder" or "decoder"
    save_file : str
        Name of the file used to save the figure.
    overwrite : bool
        If True, overwrite any existing file, else skip them.

    Returns
    -------
    None
    """
    save_file, exists = check_exists(save_file)
    if exists and overwrite is False:
        logger.info("Skipping already computed layer pair of {}".format(save_file))
        return

    df = pd.read_csv(input_file, sep="\t", header=0)
    # Keep only similarity between identical runs, epochs and models
    df = df[(df["m1_name"] == df["m2_name"]) & (df["m1_seed"] == df["m2_seed"]) & (df["p1_value"] == df["p2_value"]) &
            (df["m1_epoch"] == df["m2_epoch"]) & (df["p1_value"] == regularisation)]
    param = df["p1_name"].values[0]
    # Keep only similarity between the given encoder layer and any decoder layer
    df = df[(df["m1"] == "{}".format(layer)) & (df["m2"].str.contains("{}/[1,2,3,4,5,6]".format(target)))]
    df["m2"] = df["m2"].str.replace("{}/".format(target), "")
    df.rename(columns={"p1_value": param, "m1_epoch": "epoch", "m2": "{} layer".format(target.capitalize())}, inplace=True)
    ax = sns.lineplot(data=df, x="epoch", y=metric_name, hue="{} layer".format(target.capitalize()),
                      style="{} layer".format(target.capitalize()))
    ax.set(ylim=(0, 1))
    plt.legend(loc="center right")
    ax.legend().set_title("{} layer".format(target.capitalize()))
    save_figure(save_file)


def plot_tsne(input_dir, save_file, target, overwrite):
    """ Plot t-SNE visualisation of the similarity matrices of different models.

    Parameters
    ----------
    input_dir : str
        Absolute path to the directory where the similarity scores are stored
    save_file : str
        Name of the file used to save the figure.
    target : str
        can be "seed" or "regularisation"
    overwrite : bool
        If True, overwrite any existing file, else skip them.

    Returns
    -------
    None
    """
    save_file, exists = check_exists(save_file)
    if exists and overwrite is False:
        logger.info("Skipping already computed graph of {}".format(save_file))
        return

    files = glob("{}/*.tsv".format(input_dir))
    to_drop = ["m1_seed", "m2_seed", "m1_epoch", "m2_epoch", "p1_name", "p2_name", "p1_value", "p2_value",
               "m1_name", "m2_name"]
    X, m1_labels, m2_labels = [], [], []

    if target == "seed":
        l1, l2 = "m1_seed", "m1_epoch"
        hue, style = "Seed", "Epoch"
    elif target == "regularisation":
        l1, l2 = "p1_value", "m1_epoch"
        hue, style = "Regularisation", "Epoch"
    elif target == "classification":
        l1, l2 = "m2_name", "p2_value"
        hue, style = "Model name", "Regularisation"
    else:
        raise NotImplementedError("Target {} is not implemented. Available targets are seed, regularisation and "
                                  "classification".format(target))
    for f in files:
        df = pd.read_csv(f, sep="\t", header=0, index_col=0)
        if target != "classification":
            df = df[(df["m1_name"] == df["m2_name"]) & (df["m1_epoch"] == df["m2_epoch"]) & (df["p1_value"] == df["p2_value"])]
        else:
            df = df[(df["m1_epoch"] == df["m2_epoch"])]
        if not df.empty:
            m1_labels.append(df[l1].values[0])
            m2_labels.append(df[l2].values[0])
            x = df.drop(columns=to_drop).to_numpy().ravel()
            # Keep the lower triangular part of the vectorised symmetric matrix
            x = x[:x.shape[0] // 2]
            X.append(x)
    X = np.array(X)
    X = PCA(n_components=50).fit_transform(X)
    X = TSNE(random_state=0).fit_transform(X)
    df = pd.DataFrame.from_dict({"t-SNE dim 1": X[:, 0], "t-SNE dim 2": X[:, 1], hue: m1_labels, style: m2_labels})
    sns.scatterplot(x="t-SNE dim 1", y="t-SNE dim 2", hue=hue, style=style, data=df)
    save_figure(save_file)


def aggregate_similarity(metric_name, input_dir, save_file, overwrite):
    """ Aggregate similarity results in `input_dir` and save the results to `save_file` in TSV format.

    Parameters
    ----------
    metric_name : str
        Name of the similarity metric used
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
    files = glob("{}/*.tsv".format(input_dir))
    logger.debug("Files to process are {}".format(files))
    cleaned_dfs = []
    dfs = [pd.read_csv(f, sep="\t", header=0, index_col=0) for f in files]
    logger.info("Preparing dataframes for aggregation...")
    for df in dfs:
        logger.debug("Processing dataframe:\n{}".format(df))
        df = df.reset_index()
        df = df.melt(id_vars=["m1_seed", "m2_seed", "m1_epoch", "m2_epoch", "p1_name", "p2_name", "p1_value",
                              "p2_value", "m2", "m1_name", "m2_name"],
                     var_name="m1", value_name=metric_name)
        logger.debug("Cleaned dataframe:\n{}".format(df))
        cleaned_dfs.append(df)
    logger.info("Aggregating the dataframes...")
    df = pd.concat(cleaned_dfs)
    df.to_csv(save_file, sep="\t", index=False)
