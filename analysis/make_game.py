"""Plots the values and advantages on the video of the game;
This requires the saved model of the agent inside the run as agent.pkl

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

from train import DuelConvModel, get_options, Agent, _get_net, eval
from src.crafter_wrapper import Env
from torch import Tensor
from typing import Tuple


class EvalAgent(Agent):
    def __init__(self, estimator: nn.Module) -> None:
        self._estimator = estimator.eval()

    @torch.no_grad()
    def act(self, state: Tensor) -> int:
        return self._estimator(state).argmax(dim=1).item()


if __name__ == "__main__":
    opt = get_options(False)
    opt.video = True

    filenames = sorted(list(pathlib.Path(opt.logdir).glob(f"**/*/agent.pkl")))

    opt.logdir = opt.logdir + "/games"

    env = Env("eval", opt)
    net = _get_net(opt, env)
    agent = EvalAgent(net)

    for filename in filenames:
        net.load_state_dict(torch.load(filename))

        eval(agent, env, 1_000_000, opt)

