import os
import re
import pickle
import pathlib

import pandas as pd

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
    """Read pkl stats file from logdir/agent/* style path

    The method will read all the stats files from the given agent. Each run
    will be identified by the run column.
    """
    filenames = sorted(list(pathlib.Path(logdir).glob(f"**/*/{mode}_stats.pkl")))
    filenames = filter(lambda f: re.match(rf"^.*/[0-9]+/.*.pkl", str(f)), filenames)
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
    """Read jsonl stats file from logdir/agent/* style path

    The method will read all the stats files from the given agent. Each run
    will be identified by the run column.
    """
    filenames = sorted(list(pathlib.Path(logdir).glob(f"**/*/{mode}/stats.jsonl")))
    filenames = filter(lambda f: re.match(rf"^.*/[0-9]+/.*.jsonl", str(f)), filenames)
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


def compute_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the success rate using the given dataframe of achievements"""
    mask = df.columns.str.startswith("achievement_")
    df = (df.loc[:, mask] > 0).sum(axis=0) / len(df)
    df = pd.DataFrame(columns=["rate"], data=df)
    df.index = df.index.str.split("achievement_").map(lambda xs: xs[-1])
    return df


def read_crafter_success(
    logdir: str, clip: bool = True, mode: str = "eval"
) -> pd.DataFrame:
    """Compute the success using the jsonl stats files"""
    df = read_crafter_record(logdir, clip, mode)

    if df is None:
        return None

    return compute_success_rate(df)


def read_success(logdir: str, mode: str = "eval") -> pd.DataFrame:
    """Read the success for each agent in the logdir folder

    Each agent is identified by the agent column.
    """
    agents = [pathlib.Path(f.path).stem for f in os.scandir(logdir) if f.is_dir()]

    if not agents:
        return None

    dfs = []
    for agent in agents:
        df = read_crafter_success(os.path.join(logdir, agent), mode=mode)

        if df is None:
            print(f"Warning: Agent {agent} is missing success logs")
            continue

        df["agent"] = agent
        dfs.append(df)

    return pd.concat(dfs)


def read_logs(logdir: str, mode: str = "eval") -> pd.DataFrame:
    """Read the logs for each agent in the logdir folder

    Each agent is identified by the agent column.
    """
    agents = [pathlib.Path(f.path).stem for f in os.scandir(logdir) if f.is_dir()]

    if not agents:
        return None

    dfs = []
    for agent in agents:
        df = read_crafter_logs(os.path.join(logdir, agent), mode=mode)
        if df is None:
            print(f"Warning: Agent {agent} is missing logs")
            continue

        df["agent"] = agent
        dfs.append(df)

    return pd.concat(dfs)
