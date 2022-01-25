import itertools
import seaborn as sns
from glob import glob
import pandas as pd
from vae_ld.visualisation import logger
from vae_ld.visualisation.utils import save_figure
import matplotlib.pyplot as plt

sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})


def cka_heatmap(input_file):
    df = pd.read_csv(input_file, sep="\t", header=0)
    col_order = (["encoder/{}".format(i) for i in range(1, 7)] + ["encoder/z_mean", "encoder/z_log_var", "sampling"] +
                 ["decoder/{}".format(i) for i in range(1, 7)])
    df = df[~(df["m1"].isin(["decoder/reshape", "decoder/output"])) &
            ~(df["m2"].isin(["decoder/reshape", "decoder/output"]))]
    params = itertools.product(df["m1_name"].unique().tolist(), df["m2_name"].unique().tolist(),
                               df["p1_value"].unique().tolist(), df["p2_value"].unique().tolist(),
                               df["m1_seed"].unique().tolist(), df["m2_seed"].unique().tolist(),
                               df["m1_epoch"].unique().tolist(), df["m2_epoch"].unique().tolist())
    for m1n, m2n, p1, p2, s1, s2, e1, e2 in params:
        df2 = df[(df["p1_value"] == p1) & (df["p2_value"] == p2) & (df["m1_seed"] == s1) & (df["m2_seed"] == s2)
                 & (df["m1_epoch"] == e1) & (df["m2_epoch"] == e2)]
        if df2.empty:
            continue
        logger.info("Computing heatmap of {}, param={}, seed={} and {}, param={}, seed={}".format(m1n, p1, s1, m2n, p2, s2))
        ax = sns.heatmap(df2.pivot("m1", "m2", "cka").reindex(index=col_order, columns=col_order))
        ax.set(ylabel="{}, {}={}, seed={}, epoch={}".format(m1n, df2["p1_name"].values[0], p1, s1, e1),
               xlabel="{}, {}={}, seed={}, epoch={}".format(m2n, df2["p2_name"].values[0], p2, s2, e2))
        save_figure("{}_{}_seed_{}_epoch_{}_{}_{}_seed_{}_epoch_{}.pdf".format(m1n, p1, s1, e1, m2n, p2, s2, e2))


def avg_cka_layer_pair(input_file, m1_layer, m2_layer, save_file):
    """ Returns a lines plot of the average CKA values between two layers
    over different epochs with different regularisation strength (one line per regularisation weight).
    """
    save_file = save_file.replace("/", "_")
    df = pd.read_csv(input_file, sep="\t", header=0)
    # Keep only CKA between identical runs, epochs and models
    df = df[(df["m1_name"] == df["m2_name"]) & (df["m1_seed"] == df["m2_seed"]) & (df["p1_value"] == df["p2_value"])
            & (df["m1_epoch"] == df["m2_epoch"])]
    param = df["p1_name"].values[0]
    # Keep only CKA between m1 and m2 layers
    df = df[(df["m1"] == m1_layer) & (df["m2"] == m2_layer)]
    df.rename(columns={"p1_value": param, "m1_epoch": "epoch"}, inplace=True)
    sns.lineplot(data=df, x="epoch", y="cka", hue=param, style=param)
    save_figure(save_file)


def avg_cka_layer_list(input_file, regularisation, layer, target, save_file):
    """ Returns a lines plot of the average CKA values between the mean layer and each decoder layers
    over different epochs (one line per sampled-decoder similarity score).
    """
    save_file = save_file.replace("/", "_")
    df = pd.read_csv(input_file, sep="\t", header=0)
    # Keep only CKA between identical runs, epochs and models
    df = df[(df["m1_name"] == df["m2_name"]) & (df["m1_seed"] == df["m2_seed"]) & (df["p1_value"] == df["p2_value"]) &
            (df["m1_epoch"] == df["m2_epoch"]) & (df["p1_value"] == regularisation)]
    param = df["p1_name"].values[0]
    # Keep only CKA between the given encoder layer and any decoder layer
    df = df[(df["m1"] == "{}".format(layer)) & (df["m2"].str.contains("{}/[1,2,3,4,5,6]".format(target)))]
    df["m2"] = df["m2"].str.replace("{}/".format(target), "")
    df.rename(columns={"p1_value": param, "m1_epoch": "epoch", "m2": "{} layer".format(target.capitalize())}, inplace=True)
    sns.lineplot(data=df, x="epoch", y="cka", hue="{} layer".format(target.capitalize()),
                 style="{} layer".format(target.capitalize()))
    plt.legend(loc="center right")
    save_figure(save_file)


def aggregate_cka(input_dir, save_file):
    """ Aggregate CKA results per seed and regularisation strength
    """
    files = glob("{}/*.tsv".format(input_dir))
    cleaned_dfs = []
    dfs = [pd.read_csv(f, sep="\t", header=0, index_col=0) for f in files]
    logger.info("Preparing dataframes for aggregation...")
    for df in dfs:
        df = df.reset_index()
        df = df.melt(id_vars=["m1_seed", "m2_seed", "m1_epoch", "m2_epoch", "p1_name", "p2_name", "p1_value",
                              "p2_value", "m2", "m1_name", "m2_name"],
                     var_name="m1", value_name="cka")
        cleaned_dfs.append(df)
    logger.info("Aggregating the dataframes...")
    df = pd.concat(cleaned_dfs)
    df.to_csv(save_file, sep="\t", index=False)
