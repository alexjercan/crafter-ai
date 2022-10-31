import gym
import pathlib
from collections import deque

import crafter
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple

BoxSpace = gym.spaces.Box


class Env:
    def __init__(self, mode, args):
        assert mode in (
            "train",
            "eval",
        ), "`mode` argument can either be `train` or `eval`"
        self.device = args.device
        env = crafter.Env()
        if mode == "train":
            env = crafter.Recorder(
                env,
                pathlib.Path(args.logdir) / "train",
                save_stats=True,
                save_video=False,
                save_episode=False,
            )
        if mode == "eval":
            env = crafter.Recorder(
                env,
                pathlib.Path(args.logdir) / "eval",
                save_stats=True,
                save_video=args.video,
                save_episode=False,
            )
        self._obs_dim = 64
        self.env = env
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)

    @property
    def observation_space(self):
        return BoxSpace(0, 1, (self._obs_dim, self._obs_dim), np.float32)

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        for _ in range(self.window):
            self.state_buffer.append(
                torch.zeros(self._obs_dim, self._obs_dim, device=self.device)
            )
        obs = self.env.reset()
        obs = obs.mean(-1)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.mean(-1)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0), reward, done, info


class ReplayEnv:
    def __init__(self, replaydir: str):
        self.replaydir = replaydir
        self.filenames = sorted(list(pathlib.Path(replaydir).glob(f"**/*.npz")))

        self.file_idx = 0
        self.data_idx = 0

    def reset(self):
        data = np.load(self.filenames[self.file_idx], allow_pickle=True)
        self.images = data["image"]
        self.actions = data["action"]
        self.rewards = data["reward"]
        self.dones = data["done"]

        self.file_idx += 1
        self.data_idx = 1

        return self.images[self.data_idx - 1, ...]

    def get_action(self):
        return self.actions[self.data_idx]

    def step(self, _):
        self.data_idx += 1

        return (
            self.images[self.data_idx - 1, ...],
            self.rewards[self.data_idx - 1],
            self.dones[self.data_idx - 1],
            None,
        )


def human_to_buffer(
    replaydir: str, history_length: int
) -> List[Tuple[np.array, int, float, np.array, bool]]:
    class Scuff:
        def __init__(self, history_length):
            self.logdir = "logdir"
            self.history_length = history_length
            self.device = "cpu"
            self.video = False

    args = Scuff(history_length)

    env = Env("train", args)
    env.env = ReplayEnv(replaydir)

    state = env.reset()

    states = []
    actions = []
    rewards = []
    states_next = []
    dones = []
    while True:
        action = env.env.get_action()
        state_, reward, done, _ = env.step(action)

        states.append(state.cpu())
        actions.append(action)
        rewards.append(reward)
        states_next.append(state_.cpu())
        dones.append(bool(done))

        if done and env.env.file_idx >= len(env.env.filenames):
            break

        if done:
            state = env.reset()

    return list(zip(states, actions, rewards, states_next, dones))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/human_agent",
        help="Path to the folder containing different runs.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        dest="history_length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    cfg = parser.parse_args()

    buffer = human_to_buffer(cfg.logdir, cfg.history_length)

    states, actions, rewards, states_next, dones = zip(*buffer)

    data = {
        "state": torch.concat(states, dim=0).numpy(),
        "action": np.array(actions),
        "reward": np.array(rewards),
        "next_state": torch.concat(states_next, dim=0).numpy(),
        "done": np.array(dones),
    }

    np.savez_compressed("replay.npz", **data)

