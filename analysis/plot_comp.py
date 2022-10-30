import os
import pathlib
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_stats import read_crafter_logs, read_crafter_record, compute_success_rate
from typing import List


def read_success(logdir: str, mode: str = "eval") -> pd.DataFrame:
    agents = [pathlib.Path(f.path).stem for f in os.scandir(logdir) if f.is_dir()]

    if not agents:
        return None

    dfs = []
    for agent in agents:
        df = read_crafter_record(os.path.join(logdir, agent), mode=mode)
        df = compute_success_rate(df)
        df["agent"] = agent
        dfs.append(df)

    return pd.concat(dfs)


def read_logs(logdir: str, mode: str = "eval") -> pd.DataFrame:
    agents = [pathlib.Path(f.path).stem for f in os.scandir(logdir) if f.is_dir()]

    if not agents:
        return None

    dfs = []
    for agent in agents:
        df = read_crafter_logs(os.path.join(logdir, agent), mode=mode)
        df["agent"] = agent
        dfs.append(df)

    return pd.concat(dfs)


def plot_comp(logdir: str) -> None:
    eval_success_df = read_success(logdir, mode="eval")

    # plot success rate
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

    eval_df = read_logs(logdir, mode="eval")

    # plot eval average return
    if eval_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(x="step", y="avg_return", hue="agent", data=eval_df, ax=ax, errorbar=("se", 2))
        ax.set_xlabel("step")
        ax.set_ylabel("avg return")
        fig.suptitle("Eval Average Return")
        fig.savefig(
            pathlib.Path(logdir) / "eval_average_return.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/",
        help="Path to the folder containing different runs.",
    )
    cfg = parser.parse_args()

    plot_comp(cfg.logdir)
