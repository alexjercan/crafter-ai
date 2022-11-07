"""This plots the stats for all the agents in the logdir

For now this plots:
    - The average return of each agent
    - The success rate of each agent
"""
import pathlib
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

from common import read_logs, read_success


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
        sns.lineplot(
            x="step",
            y="avg_return",
            hue="agent",
            data=eval_df,
            ax=ax,
            errorbar=("se", 2),
        )
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
