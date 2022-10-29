import gym
import pathlib
from collections import deque

import crafter
import numpy as np
import torch
from PIL import Image

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
                pathlib.Path(args.logdir),
                save_stats=True,
                save_video=False,
                save_episode=False,
            )
        if mode == "eval":
            env = crafter.Recorder(
                env,
                pathlib.Path(args.logdir) / "video",
                save_stats=False,
                save_video=True,
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
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.mean(-1)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0), reward, done, info
