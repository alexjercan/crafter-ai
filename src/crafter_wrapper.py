import pathlib
from collections import deque

import crafter
import numpy as np
import torch
from PIL import Image


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
        self._obs_dim = 64
        # env = ResizeImage(env)
        env = GrayScale(env)
        self.env = env
        self.action_space = env.action_space
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)

    def reset(self):
        for _ in range(self.window):
            self.state_buffer.append(
                torch.zeros(self._obs_dim, self._obs_dim, device=self.device)
            )
        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0), reward, done, info


class GrayScale:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = obs.mean(-1)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = obs.mean(-1)
        return obs


class ResizeImage:
    def __init__(self, env, size=48):
        self._size = size
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._resize(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = self._resize(obs)
        return obs

    def _resize(self, image):
        print(image.shape)
        image = Image.fromarray(image)
        image = image.resize((self._size, self._size), Image.NEAREST)
        image = np.array(image)
        return image
