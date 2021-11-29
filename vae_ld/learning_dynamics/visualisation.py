import seaborn as sns
from glob import glob
import pandas as pd
from pathlib import Path
from vae_ld.learning_dynamics.utils import save_figure


def cka_heatmap(cfg):
    sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})
    files = glob("{}/*.tsv".format(cfg.input_dir))
    dfs = [(Path(f).stem, pd.read_csv(f, sep="\t", header=0, index_col=0)) for f in files]
    for f, df in dfs:
        save_path = "{}.pdf".format(f)
        ax = sns.heatmap(df)
        ax.set(xlabel="{}, {}={}, seed={}".format(cfg.m1_name, cfg.p1_name, cfg.p1_value, cfg.m1_seed),
               ylabel="{}, {}={}, seed={}".format(cfg.m2_name, cfg.p2_name, cfg.p2_value, cfg.m2_seed))
        save_figure(save_path)
