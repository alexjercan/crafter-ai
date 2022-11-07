"""Plots the stats of an agent using the runs

For now it plots:
    - The avg return
    - The train loss
    - The qsa values for train
    - Success rate for the runs
"""
import pathlib
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

from common import read_crafter_logs, read_crafter_success


def plot_stats(logdir: str) -> None:
    eval_df = read_crafter_logs(logdir, mode="eval")

    # plot eval average return
    if eval_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(x="step", y="avg_return", data=eval_df, ax=ax, errorbar=("se", 2))
        ax.set_xlabel("step")
        ax.set_ylabel("avg return")
        fig.suptitle("Eval Average Return")
        fig.savefig(
            pathlib.Path(logdir) / "eval_average_return.png",
            dpi=300,
            bbox_inches="tight",
        )

    train_df = read_crafter_logs(logdir, mode="train")

    # plot train loss
    if train_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(x="step", y="loss", data=train_df, ax=ax, errorbar=("se", 2))
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        fig.suptitle("Train Loss")
        fig.savefig(
            pathlib.Path(logdir) / "train_loss.png", dpi=300, bbox_inches="tight"
        )

    # plot train qsa
    if train_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(
            x="step", y="avg_qsa", data=train_df, label="qsa", ax=ax, errorbar=("se", 2)
        )
        sns.lineplot(
            x="step",
            y="avg_target_qsa",
            data=train_df,
            label="target_qsa",
            ax=ax,
            errorbar=("se", 2),
        )
        ax.set_xlabel("step")
        ax.set_ylabel("q value")
        fig.suptitle("Q Value Function")
        fig.savefig(
            pathlib.Path(logdir) / "train_qsa.png", dpi=300, bbox_inches="tight"
        )

    eval_success_df = read_crafter_success(logdir, mode="eval")

    # plot success rate
    if eval_success_df is not None:
        fig, ax = plt.subplots()
        sns.barplot(x=eval_success_df.index, y=eval_success_df["rate"], ax=ax)
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
        default="logdir/random_agent",
        help="Path to the folder containing different runs.",
    )
    cfg = parser.parse_args()

    plot_stats(cfg.logdir)
