"""Plots the values and advantages on the video of the game; This requires the saved model of the agent inside the run as agent.pkl

This creates a video with the gameplay and the values as red on top
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import PIL
import torch
import pathlib
import imageio

import numpy as np
import torch.nn as nn

from train import DuelConvModel, get_options, DQNAgent, Agent
from src.crafter_wrapper import Env
from torch import Tensor
from typing import Tuple


class DuelConvModelMod(DuelConvModel):
    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = x.reshape(x.shape[0], -1)

        values = self.value(x)
        values_ = self.value_c(values)
        advantages = self.advantage(x)
        advantages_ = self.advantage_c(advantages)

        advantages_ = advantages_ - advantages_.mean(dim=1, keepdim=True)

        return values_ + advantages_, values, advantages


class EvalAgent(Agent):
    def __init__(self, estimator: nn.Module) -> None:
        self._estimator = estimator

    @torch.no_grad()
    def act(self, state: Tensor) -> int:
        (q_values, values, advantages) = self._estimator(state)
        return q_values.argmax(dim=1).item(), values, advantages


def to_rgb_red(xs: np.ndarray, size: Tuple[float, float] = (512, 512)) -> np.ndarray:
    xs = np.array(PIL.Image.fromarray(xs).resize(size))
    ys = np.zeros((*size, 3))
    ys[:, :, 0] = xs

    return np.uint8(ys * 255)


def eval_model(path: str, logdir: str, env: Env, net: DuelConvModelMod) -> None:
    net.load_state_dict(torch.load(path))
    agent = EvalAgent(net)

    episode_values = []
    episode_advantages = []

    state, done = env.reset(), False

    while not done:
        action, values, advantages = agent.act(state)
        state, reward, done, info = env.step(action)

        episode_values.append(to_rgb_red(values.reshape(16, 16).numpy()))
        episode_advantages.append(to_rgb_red(advantages.reshape(16, 16).numpy()))

    def comb(f: np.ndarray, v: np.ndarray) -> np.ndarray:
        r = (f[:, :, 0] + v[:, :, 0] // 2)
        g = f[:, :, 1]
        b = f[:, :, 2]

        return np.stack([r, g, b], axis=2)

    frames_values = list(map(comb, env.env._frames, episode_values))
    imageio.mimsave(logdir + "/values.mp4", frames_values)

    frames_advantages = list(map(comb, env.env._frames, episode_advantages))
    imageio.mimsave(logdir + "/advantages.mp4", frames_advantages)


if __name__ == "__main__":
    opt = get_options(False)
    opt.video = True

    filenames = sorted(list(pathlib.Path(opt.logdir).glob(f"**/*/agent.pkl")))

    opt.logdir = opt.logdir + "/analysis"

    env = Env("eval", opt)
    net = DuelConvModelMod(
        opt.history_length, env.action_space.n, env.observation_space.shape
    ).to(opt.device)

    for filename in filenames:
        eval_model(filename, opt.logdir, env, net)
