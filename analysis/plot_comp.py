import os
import pathlib
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List


def read_success(logdir: str, mode: str = "eval") -> pd.DataFrame:
    agents = [pathlib.Path(f.path).stem for f in os.scandir(logdir) if f.is_dir()]

    if not agents:
        return None

    dfs = []
    for agent in agents:
        df = pd.read_csv(
            pathlib.Path(logdir) / agent / f"{mode}_success.csv", index_col=0
        )
        df.index = df.index.str.split("achievement_").map(lambda xs: xs[-1])
        df["agent"] = agent
        dfs.append(df)

    return pd.concat(dfs)


def plot_comp(logdir: str) -> None:
    eval_success_df = read_success(logdir, mode="eval")

    if eval_success_df is not None:
        fig, ax = plt.subplots()
        sns.barplot(
            x=eval_success_df.index,
            y=eval_success_df["rate"],
            hue=eval_success_df["agent"],
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel("achievements")
        ax.set_ylabel("success rate")
        fig.suptitle("Eval Success Rate")
        fig.savefig(
            pathlib.Path(logdir) / "eval_success.png", dpi=300, bbox_inches="tight"
        )
        eval_success_df.to_csv(pathlib.Path(logdir) / "eval_success.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/",
        help="Path to the folder containing different runs.",
    )
    cfg = parser.parse_args()

    plot_comp(cfg.logdir)
