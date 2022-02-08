import pathlib

import seaborn as sns
from glob import glob
import pandas as pd
from vae_ld.visualisation import logger
from vae_ld.visualisation.utils import save_figure
import matplotlib.pyplot as plt

sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})


def similarity_heatmap(metric_name, input_file, overwrite):
    df = pd.read_csv(input_file, sep="\t", header=0)
    grouped_df = df.groupby(["m1_name", "p1_value", "m1_seed", "m1_epoch", "m2_name", "p2_value", "m2_seed",
                             "m2_epoch"])
    group_names = grouped_df.groups.keys()
    col_order = (["input"] + ["encoder/{}".format(i) for i in range(1, 7)] +
                 ["encoder/z_mean", "encoder/z_log_var", "sampling"] + ["decoder/{}".format(i) for i in range(1, 7)])

    for group_name in group_names:
        cfg = "{}, param={}, seed={}, epoch={} and {}, param={}, seed={}, epoch={}".format(*group_name)
        save_path = "{}_{}_seed_{}_epoch_{}_{}_{}_seed_{}_epoch_{}.pdf".format(*group_name)

        if overwrite is False and pathlib.Path(save_path).exists():
            logger.info("Skipping already plotted heatmap of {}".format(cfg))
            continue

        group = grouped_df.get_group(group_name)
        logger.info("Plotting heatmap of {}".format(cfg))
        ax = sns.heatmap(group.pivot("m1", "m2", metric_name).reindex(index=col_order, columns=col_order), vmin=0, vmax=1)
        ax.set(ylabel="{}, {}={}, seed={}, epoch={}".format(group_name[0], group["p1_name"].values[0], *group_name[1:4]),
               xlabel="{}, {}={}, seed={}, epoch={}".format(group_name[4], group["p2_name"].values[0], *group_name[5:]))
        save_figure(save_path)

    # col_order = (["encoder/{}".format(i) for i in range(1, 7)] + ["encoder/z_mean", "encoder/z_log_var", "sampling"] +
    #              ["decoder/{}".format(i) for i in range(1, 7)])
    # df = df[~(df["m1"].isin(["decoder/reshape", "decoder/output"])) &
    #         ~(df["m2"].isin(["decoder/reshape", "decoder/output"]))]
    # params = itertools.product(df["m1_name"].unique().tolist(), df["m2_name"].unique().tolist(),
    #                            df["p1_value"].unique().tolist(), df["p2_value"].unique().tolist(),
    #                            df["m1_seed"].unique().tolist(), df["m2_seed"].unique().tolist(),
    #                            df["m1_epoch"].unique().tolist(), df["m2_epoch"].unique().tolist())
    #
    # for m1n, m2n, p1, p2, s1, s2, e1, e2 in params:
    #     cfg = "{}, param={}, seed={}, epoch={} and {}, param={}, seed={}, epoch={}".format(m1n, p1, s1, e1, m2n, p2, s2,
    #                                                                                        e2)
    #     save_path = "{}_{}_seed_{}_epoch_{}_{}_{}_seed_{}_epoch_{}.pdf".format(m1n, p1, s1, e1, m2n, p2, s2, e2)
    #
    #     if pathlib.Path(save_path).exists() and overwrite is False:
    #         logger.info("Skipping already computed heatmap of {}".format(cfg))
    #         continue
    #
    #     df2 = df[(df["m1_name"] == m1n) & (df["m2_name"] == m2n) & (df["p1_value"] == p1) & (df["p2_value"] == p2)
    #              & (df["m1_seed"] == s1) & (df["m2_seed"] == s2) & (df["m1_epoch"] == e1) & (df["m2_epoch"] == e2)]
    #     if df2.empty:
    #         logger.info("Skipping empty config of {}".format(cfg))
    #         continue
    #
    #     logger.info("Computing heatmap of {}".format(cfg))
    #     ax = sns.heatmap(df2.pivot("m1", "m2", metric_name).reindex(index=col_order, columns=col_order), vmin=0, vmax=1)
    #     ax.set(ylabel="{}, {}={}, seed={}, epoch={}".format(m1n, df2["p1_name"].values[0], p1, s1, e1),
    #            xlabel="{}, {}={}, seed={}, epoch={}".format(m2n, df2["p2_name"].values[0], p2, s2, e2))
    #     save_figure(save_path)


def avg_similarity_layer_pair(metric_name, input_file, m1_layer, m2_layer, save_file, overwrite):
    """ Returns a lines plot of the average similarity values between two layers
    over different epochs with different regularisation strength (one line per regularisation weight).
    """
    save_file = save_file.replace("/", "_")
    if pathlib.Path(save_file).exists() and overwrite is False:
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
    sns.lineplot(data=df, x="epoch", y=metric_name, hue=param, style=param)
    save_figure(save_file)


def avg_similarity_layer_list(metric_name, input_file, regularisation, layer, target, save_file, overwrite):
    """ Returns a lines plot of the average similarity values between the mean layer and each decoder layers
    over different epochs (one line per sampled-decoder similarity score).
    """
    save_file = save_file.replace("/", "_")
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already computed layer list of {}".format(save_file))
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
    sns.lineplot(data=df, x="epoch", y=metric_name, hue="{} layer".format(target.capitalize()),
                 style="{} layer".format(target.capitalize()))
    plt.legend(loc="center right")
    save_figure(save_file)


def aggregate_similarity(metric_name, input_dir, save_file, overwrite):
    """ Aggregate similarity results per seed and regularisation strength
    """
    if pathlib.Path(save_file).exists() and overwrite is False:
        logger.info("Skipping already computed aggregation of {}".format(save_file))
        return
    files = glob("{}/*.tsv".format(input_dir))
    cleaned_dfs = []
    dfs = [pd.read_csv(f, sep="\t", header=0, index_col=0) for f in files]
    logger.info("Preparing dataframes for aggregation...")
    for df in dfs:
        df = df.reset_index()
        df = df.melt(id_vars=["m1_seed", "m2_seed", "m1_epoch", "m2_epoch", "p1_name", "p2_name", "p1_value",
                              "p2_value", "m2", "m1_name", "m2_name"],
                     var_name="m1", value_name=metric_name)
        cleaned_dfs.append(df)
    logger.info("Aggregating the dataframes...")
    df = pd.concat(cleaned_dfs)
    df.to_csv(save_file, sep="\t", index=False)
