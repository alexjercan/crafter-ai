import torch

import numpy as np
import torch.optim as optim
import torch.nn as nn

from .crafter_wrapper import Env
from torch import Tensor
from typing import Union, Tuple


class ReplayBuffer(object):
    def __init__(self, size: int = 10000):
        self.size = size
        self.length = 0
        self.idx = -1

        self.states = None
        self.states_next = None
        self.actions = None
        self.rewards = None
        self.done = None

    def store(
        self,
        s: Tensor,
        a: int,
        r: float,
        s_next: Tensor,
        done: bool,
    ):

        if self.states is None:
            self.states = torch.zeros([self.size] + list(s.shape))
            self.states_next = torch.zeros_like(self.states)
            self.actions = torch.zeros((self.size,))
            self.rewards = torch.zeros((self.size,))
            self.done = torch.zeros((self.size,))

        self.idx = (self.idx + 1) % self.size
        self.length = min(self.length + 1, self.size)
        self.states[self.idx] = s.clone()
        self.actions[self.idx] = torch.tensor(a)
        self.rewards[self.idx] = torch.tensor(r)
        self.done[self.idx] = torch.tensor(done)
        self.states_next[self.idx] = s_next.clone()

    def sample(
        self, batch_size: int = 128
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert self.length >= batch_size, "Can not sample from the buffer yet"
        indices = np.random.choice(
            a=np.arange(self.length), size=batch_size, replace=False
        )

        s = self.states[indices]
        s_next = self.states_next[indices]
        a = self.actions[indices]
        r = self.rewards[indices]
        done = self.done[indices]

        return s, a, r, s_next, done


def select_epsilon_greedy_action(env: Env, Q: nn.Module, s: Tensor, eps: float):
    rand = np.random.rand()

    if rand < eps:
        return env.action_space.sample()

    with torch.no_grad():
        s = s[None].to(env.device)
        output = Q(s).argmax(dim=1).item()

    return output


def init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def linear_eps_generator(start: float = 1.0, end: float = 0.1, num_steps: int = 10000):
    crt_iter = -1

    while True:
        crt_iter += 1
        frac = min(crt_iter / num_steps, 1)
        eps = (1 - frac) * start + frac * end
        yield eps


@torch.no_grad()
def dqn_target(
    Q: nn.Module,
    target_Q: nn.Module,
    r_batch: Tensor,
    s_next_batch: Tensor,
    done_batch: Tensor,
    gamma: float,
) -> Tensor:
    next_Q_values = target_Q(s_next_batch).max(dim=1)[0]
    next_Q_values[done_batch == 1] = 0
    return r_batch + (gamma * next_Q_values)


@torch.no_grad()
def ddqn_target(
    Q: nn.Module,
    target_Q: nn.Module,
    r_batch: Tensor,
    s_next_batch: Tensor,
    done_batch: Tensor,
    gamma: float,
) -> Tensor:
    next_Q_values = (
        target_Q(s_next_batch)
        .gather(1, Q(s_next_batch).argmax(dim=1, keepdim=True))
        .squeeze()
    )
    next_Q_values[done_batch == 1] = 0

    return r_batch + (gamma * next_Q_values)
