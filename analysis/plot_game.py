import pathlib
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from common import read_crafter_logs, read_crafter_success
import crafter


def plot_one_game(filename: str) -> None:
    data = np.load(filename, allow_pickle=True)

    actions = data["action"]
    action_names = crafter.constants.actions

    if actions is not None:
        fig, ax = plt.subplots()
        ax.hist(actions, bins=np.arange(len(action_names)) - 0.5)

        ax.set_xticks(range(len(action_names)))
        ax.set_xticklabels(action_names, rotation=90)
        ax.set_xlabel("actions")
        ax.set_ylabel("count")
        fig.suptitle("Actions Distribution")
        fig.savefig(
            pathlib.Path(filename).with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_game(logdir: str) -> None:
    filenames = sorted(list(pathlib.Path(logdir).glob(f"**/*/*.npz")))

    for filename in filenames:
        plot_one_game(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/random_agent",
        help="Path to the folder containing different runs.",
    )
    cfg = parser.parse_args()

    plot_game(cfg.logdir)
