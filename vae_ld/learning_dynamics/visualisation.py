import seaborn as sns
from glob import glob
import pandas as pd
from pathlib import Path
from vae_ld.learning_dynamics.utils import save_figure


def cka_heatmap(input_dir):
    sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})
    files = glob("{}/*.tsv".format(input_dir))
    dfs = [(Path(f).stem, pd.from_csv(f, sep="\t")) for f in files]
    for f, df in dfs:
        save_path = "{}.pdf".format(f)
        sns.heatmap(df)
        save_figure(save_path)
