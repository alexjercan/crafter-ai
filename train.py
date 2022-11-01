import os
import torch
import pathlib
import argparse
import itertools
import pickle
import random

import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm.auto import tqdm
from pathlib import Path
from src.crafter_wrapper import Env, human_to_buffer
from torch import Tensor
from typing import Iterator, Tuple, List


@dataclass
class Options:
    logdir: str
    steps: int
    history_length: int
    eval_episodes: int
    eval_interval: int
    video: bool
    device: str
    agent: str


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        after: List[str] = None,
    ):
        super().__init__()

        def str_to_block(name: str):
            if name == "bn":
                return nn.BatchNorm2d(out_channels)
            if name == "relu":
                return nn.ReLU()

            raise NotImplementedError(name)

        if after is None:
            after = []

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
            *[str_to_block(name) for name in after],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvModel(nn.Module):
    def __init__(
        self, in_features: int, num_actions: int, image_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions

        blocks = [
            ConvBlock(in_features, 16, (3, 3), padding=1, after=["relu"]),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(16, 16, (3, 3), padding=1, after=["relu"]),
            nn.MaxPool2d(3, stride=2, padding=1),
        ]

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(16 * (image_size[0] // 4) * (image_size[1] // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


class DuelConvModel(nn.Module):
    def __init__(
        self, in_features: int, num_actions: int, image_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions

        blocks = [
            ConvBlock(in_features, 16, (3, 3), padding=1, after=["relu"]),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(16, 16, (3, 3), padding=1, after=["relu"]),
            nn.MaxPool2d(3, stride=2, padding=1),
        ]

        self.blocks = nn.Sequential(*blocks)
        self.value = nn.Sequential(
            nn.Linear(16 * (image_size[0] // 4) * (image_size[1] // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(16 * (image_size[0] // 4) * (image_size[1] // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = x.reshape(x.shape[0], -1)

        values = self.value(x)
        advantages = self.advantage(x)

        advantages = advantages - advantages.mean(dim=1, keepdim=True)

        return values + advantages


class ReplayMemory:
    """Cyclic buffer that stores the transitions of the game on CPU RAM."""

    def __init__(self, opt: Options, size: int = 1000, batch_size: int = 32):
        self._buffer: deque = deque(maxlen=size)
        self._batch_size = batch_size
        self._device = opt.device

    def push(self, transition: Tuple[Tensor, int, float, Tensor, bool]) -> None:
        """Store the transition in the buffer

        The first element of the transition is the current state with the shape
        (1, ...), the second element will be the action that the agent took,
        the third will be the reward, the fourth will be the next state with
        the shame shape as the current state, and the last will be a boolen
        that will tell if the game is done.

        The function will move the tensors to cpu.
        """
        s, a, r, s_, d = transition
        self._buffer.append((s.cpu(), a, r, s_.cpu(), d))

    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample from self._buffer

        Should return a tuple of tensors of size:
        (
            states:     N , ...,
            actions:    N * 1, (torch.int64)
            rewards:    N * 1, (torch.float32)
            states_:    N * ...,
            done:       N * 1, (torch.uint8)
        )

        where N is the batch_size.
        """
        # sample
        s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))

        # reshape, convert if needed, put on device
        return (
            torch.cat(s, 0).to(self._device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self._device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self._device),
            torch.cat(s_, 0).to(self._device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(self._device),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class ExtendedMemory(ReplayMemory):
    def __init__(
        self,
        opt: Options,
        replaydir: str = "logdir/human_agent/",
        size: int = 1000,
        batch_size: int = 32,
        alpha: float = 0.5,  # the chance that we get tuple from human gameplay
    ):
        super().__init__(opt, size, batch_size)
        self._human = human_to_buffer(replaydir, opt.history_length)
        self._alpha = alpha

    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self._alpha < torch.rand(1).item():
            s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))
        else:
            s, a, r, s_, d = zip(*random.sample(self._human, self._batch_size))

        return (
            torch.cat(s, 0).to(self._device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self._device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self._device),
            torch.cat(s_, 0).to(self._device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(self._device),
        )

    def __len__(self) -> int:
        return len(self._buffer)


def get_epsilon_schedule(start=1.0, end=0.1, steps=500) -> Iterator[float]:
    """Returns either:
    - a generator of epsilon values
    - a function that receives the current step and returns an epsilon

    The epsilon values returned by the generator or function need
    to be degraded from the `start` value to the `end` within the number
    of `steps` and then continue returning the `end` value indefinetly.

    You can pick any schedule (exp, poly, etc.). I tested with linear decay.
    """
    eps_step = (start - end) / steps

    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step

    return itertools.chain(frange(start, end, eps_step), itertools.repeat(end))


class Agent:
    def __init__(self) -> None:
        pass

    def act(self, state: Tensor) -> int:
        raise NotImplementedError()

    def step(self, state: Tensor) -> int:
        raise NotImplementedError()

    def learn(
        self,
        state: Tensor,
        action: int,
        reward: float,
        state_: Tensor,
        done: bool,
        opt: Options,
    ) -> None:
        pass


class RandomAgent(Agent):
    """An example Random Agent"""

    def __init__(self, action_num: int) -> None:
        super().__init__()
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, state: Tensor) -> int:
        return self.policy.sample().item()

    def step(self, state: Tensor) -> int:
        return self.act(state)

    def learn(
        self,
        state: Tensor,
        action: int,
        reward: float,
        state_: Tensor,
        done: bool,
        opt: Options,
    ) -> None:
        pass


class DQNAgent(Agent):
    def __init__(
        self,
        estimator: nn.Module,
        buffer: ReplayMemory,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epsilon_schedule: Iterator[float],
        action_num: int,
        gamma: float = 0.92,
        update_steps: int = 4,
        update_target_steps: int = 10,
        warmup_steps: int = 100,
    ):
        super().__init__()
        self._estimator = estimator
        self._target_estimator = deepcopy(estimator)
        self._buffer = buffer
        self._criterion = criterion
        self._optimizer = optimizer
        self._epsilon = epsilon_schedule
        self._action_num = action_num
        self._gamma = gamma
        self._update_steps = update_steps
        self._update_target_steps = update_target_steps
        self._warmup_steps = warmup_steps
        self._step_cnt = 0
        assert (
            warmup_steps > self._buffer._batch_size
        ), "You should have at least a batch in the ER."

    @torch.no_grad()
    def act(self, state: Tensor) -> int:
        return self._estimator(state).argmax(dim=1).item()

    def step(self, state: Tensor) -> int:
        """Get an action based on the input state

        If the agent is still in warmup, number of played stepts is smaller
        than the number of warmup steps, then output a random action.
        Otherwise, return an action using a greedy policy.
        """
        if self._step_cnt < self._warmup_steps:
            return int(torch.randint(self._action_num, (1,)).item())

        if next(self._epsilon) < torch.rand(1).item():
            return self.act(state)

        return int(torch.randint(self._action_num, (1,)).item())

    def learn(
        self,
        state: Tensor,
        action: int,
        reward: float,
        state_: Tensor,
        done: bool,
        opt: Options,
    ) -> None:
        # add transition to the experience replay
        self._buffer.push((state.cpu(), action, reward, state_.cpu(), done))

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return

        if self._step_cnt % self._update_steps == 0:
            # sample from experience replay and do an update
            batch = self._buffer.sample()
            stats = self._update(*batch)
            _save_train_stats(*stats, self._step_cnt, opt.logdir)

        # update the target estimator
        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(self._estimator.state_dict())

        self._step_cnt += 1

    def _update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        states_: Tensor,
        done: Tensor,
    ) -> Tuple[float, Tensor, Tensor]:
        # compute the DeepQNetwork update. Carefull not to include the
        # target network in the computational graph.

        # Compute Q(s, * | θ) and Q(s', . | θ^)
        q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(states_)

        # compute Q(s, a) and max_a' Q(s', a')
        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.max(1, keepdim=True)[0]

        # compute target Q(s', a')
        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        # compute the loss and average it over the entire batch
        loss = self._criterion(qsa, target_qsa)

        # backprop and optimize
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item(), qsa.detach().cpu(), target_qsa.cpu()


class DDQNAgent(DQNAgent):
    def _update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        states_: Tensor,
        done: Tensor,
    ) -> Tuple[float, Tensor, Tensor]:
        # compute the DeepQNetwork update. Carefull not to include the
        # target network in the computational graph.

        # Compute Q(s, * | θ) and Q(s', . | θ^)
        with torch.no_grad():
            actions_ = self._estimator(states_).argmax(1, keepdim=True)
            q_values_ = self._target_estimator(states_)
        q_values = self._estimator(states)

        # compute Q(s, a)
        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.gather(1, actions_)

        # compute target Q(s', a')
        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        # compute the loss and average it over the entire batch
        loss = self._criterion(qsa, target_qsa)

        # backprop and optimize
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item(), qsa.detach().cpu(), target_qsa.cpu()


def eval(agent: Agent, env: Env, crt_step: int, opt: Options) -> None:
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        state, done = env.reset(), False
        episodic_returns.append(0.0)

        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    _save_eval_stats(episodic_returns, crt_step, opt.logdir)


def _save_eval_stats(episodic_returns: List[float], crt_step: int, path: str) -> None:
    # save the evaluation stats
    episodic_returns_ = torch.tensor(episodic_returns)
    avg_return = episodic_returns_.mean().item()
    tqdm.write(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns_.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def _save_train_stats(
    loss: float, qsa: Tensor, target_qsa: Tensor, crt_step: int, path: str
) -> None:
    avg_qsa = qsa.mean().item()
    avg_target_qsa = target_qsa.mean().item()
    if False:
        tqdm.write(
            "[{:06d}] train results: loss={:03.6f}, qsa={:03.2f} std={:03.2f} target_qsa={:03.2f} std={:03.2f}".format(
                crt_step,
                loss,
                avg_qsa,
                qsa.std().item(),
                avg_target_qsa,
                target_qsa.std().item(),
            )
        )
    with open(path + "/train_stats.pkl", "ab") as f:
        pickle.dump(
            {
                "step": crt_step,
                "loss": loss,
                "avg_qsa": avg_qsa,
                "avg_target_qsa": avg_target_qsa,
            },
            f,
        )


def _info(opt: Options) -> None:
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if any(Path(opt.logdir).iterdir()):
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},64,64),"
        + "with values between 0 and 1."
    )


def _get_memory(opt: Options) -> ReplayMemory:
    if "ext" in opt.agent:
        return ExtendedMemory(size=1_000, batch_size=32, alpha=0.5, opt=opt)

    return ReplayMemory(size=1_000, batch_size=32, opt=opt)


def _get_net(opt: Options, env: Env) -> nn.Module:
    if "duel" in opt.agent:
        return DuelConvModel(
            opt.history_length, env.action_space.n, env.observation_space.shape
        ).to(opt.device)

    return ConvModel(
        opt.history_length, env.action_space.n, env.observation_space.shape
    ).to(opt.device)


def _get_agent(opt: Options, env: Env) -> Agent:
    if opt.agent == "random":
        return RandomAgent(env.action_space.n)

    buffer = _get_memory(opt)
    net = _get_net(opt, env)

    if "ddqn" in opt.agent:
        return DDQNAgent(
            net,
            buffer,
            nn.HuberLoss(),
            optim.Adam(net.parameters(), lr=3e-4, eps=1e-5),
            get_epsilon_schedule(start=1.0, end=0.1, steps=4000),
            env.action_space.n,
            warmup_steps=100,
            update_steps=2,
        )

    if "dqn" in opt.agent:
        return DQNAgent(
            net,
            buffer,
            nn.HuberLoss(),
            optim.Adam(net.parameters(), lr=3e-4, eps=1e-5),
            get_epsilon_schedule(start=1.0, end=0.1, steps=4000),
            env.action_space.n,
            warmup_steps=100,
            update_steps=2,
        )

    raise NotImplementedError(opt.agent)


def main(opt: Options) -> None:
    _info(opt)

    env = Env("train", opt)
    eval_env = Env("eval", opt)

    agent = _get_agent(opt, env)

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    pbar = tqdm(total=opt.steps, position=0, leave=True)
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            s, done = env.reset(), False
            episode_reward = 0

        a = agent.step(s)
        s_next, r, done, info = env.step(a)
        agent.learn(s, a, r, s_next, done, opt)

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)

        episode_reward += r
        s = s_next.clone()

        pbar.set_description(
            f"[Episode {ep_cnt}]: Current reward {episode_reward:.04f}"
        )
        pbar.update(1)
        step_cnt += 1


def get_options() -> Options:
    """Configures a parser. Extend this with all the best performing hyperparameters of
    your agent as defaults.

    For devel purposes feel free to change the number of training steps and
    the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/")
    parser.add_argument(
        "--steps",
        dest="steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        dest="history_length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        dest="eval_interval",
        type=int,
        default=10_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        dest="eval_episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--video",
        dest="video",
        action="store_true",
        help="Save video of eval process",
    )
    parser.add_argument(
        "--agent",
        dest="agent",
        type=str,
        default="random",
        help="The agent to use: random, dqn, ddqn with ext and duel modifiers",
    )
    args = parser.parse_args()

    logdir = os.path.join(args.logdir, f"{args.agent}_agent")
    os.makedirs(logdir, exist_ok=True)
    run = str(len([f for f in os.scandir(logdir) if f.is_dir()]))
    logdir = os.path.join(logdir, run)
    os.makedirs(logdir, exist_ok=True)

    return Options(
        logdir=logdir,
        steps=args.steps,
        history_length=args.history_length,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        video=args.video,
        device="cuda" if torch.cuda.is_available() else "cpu",
        agent=args.agent,
    )


if __name__ == "__main__":
    main(get_options())
