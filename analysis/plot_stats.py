import argparse
import pathlib
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict


def read_pkl(path: str) -> List[Dict]:
    events = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                events.append(pickle.load(openfile))
            except EOFError:
                break
    return events


def read_crafter_logs(
    logdir: str, clip: bool = True, mode: str = "eval"
) -> pd.DataFrame:
    filenames = sorted(list(pathlib.Path(logdir).glob(f"**/*/{mode}_stats.pkl")))
    runs = []
    for idx, fn in enumerate(filenames):
        df = pd.DataFrame(data=read_pkl(str(fn)))
        df["run"] = idx
        runs.append(df)

    if not runs:
        return None

    if clip:
        min_len = min([len(run) for run in runs])
        runs = [run[:min_len] for run in runs]
        print(f"Clipped all runs to {min_len}.")

    return pd.concat(runs, ignore_index=True)


def read_crafter_record(
    logdir: str, clip: bool = True, mode: str = "eval"
) -> pd.DataFrame:
    filenames = sorted(list(pathlib.Path(logdir).glob(f"**/*/{mode}/stats.jsonl")))
    runs = []
    for idx, fn in enumerate(filenames):
        df = pd.read_json(fn, lines=True)
        df["run"] = idx
        runs.append(df)

    if not runs:
        return None

    if clip:
        min_len = min([len(run) for run in runs])
        runs = [run[:min_len] for run in runs]
        print(f"Clipped all runs to {min_len}.")

    return pd.concat(runs, ignore_index=True)


def compute_success_rate(df: pd.DataFrame) -> pd.Series:
    mask = df.columns.str.startswith("achievement_")
    return (df.loc[:, mask] > 0).sum(axis=0) / len(df)



def plot_stats(logdir: str) -> None:
    eval_df = read_crafter_logs(logdir, mode="eval")

    # plot eval average return
    if eval_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(x="step", y="avg_return", data=eval_df, ax=ax)
        ax.set_xlabel("step")
        ax.set_ylabel("avg return")
        fig.suptitle("Eval Average Return")
        fig.savefig(pathlib.Path(logdir) / "eval_average_return.png", dpi=300, bbox_inches = "tight")

    train_df = read_crafter_logs(logdir, mode="train")

    # plot train loss
    if train_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(x="step", y="loss", data=train_df, ax=ax)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        fig.suptitle("Train Loss")
        fig.savefig(pathlib.Path(logdir) / "train_loss.png", dpi=300, bbox_inches = "tight")

    # plot train qsa
    if train_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(x="step", y="qsa", data=train_df, label="qsa", ax=ax)
        sns.lineplot(x="step", y="target_qsa", data=train_df, label="target_qsa", ax=ax)
        ax.set_xlabel("step")
        ax.set_ylabel("q value")
        fig.suptitle("Q Value Function")
        fig.savefig(pathlib.Path(logdir) / "train_qsa.png", dpi=300, bbox_inches = "tight")

    eval_df = read_crafter_record(logdir, mode="eval")

    # plot success rate
    if eval_df is not None:
        eval_success_df = compute_success_rate(eval_df)
        fig, ax = plt.subplots()
        sns.barplot(x=eval_success_df.index, y=eval_success_df.values, ax=ax)
        labels = list(map(lambda a: a.split("achievement_")[-1], eval_success_df.index))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlabel("achievements")
        ax.set_ylabel("success rate")
        fig.suptitle("Eval Success Rate")
        fig.savefig(pathlib.Path(logdir) / "eval_success.png", dpi=300, bbox_inches = "tight")
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
