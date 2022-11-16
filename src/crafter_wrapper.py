import gym
import random
import pathlib
from collections import deque

import crafter
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
from crafter import constants
from crafter.objects import Player

BoxSpace = gym.spaces.Box


def player_can_place(player: Player, name: str):
    target = (player.pos[0] + player.facing[0], player.pos[1] + player.facing[1])
    material, _ = player.world[target]

    if player.world[target][1]:
        return False
    info = constants.place[name]
    if material not in info["where"]:
        return False
    if any(player.inventory[k] < v for k, v in info["uses"].items()):
        return False

    return True


def player_can_make(player: Player, name: str):
    nearby, _ = player.world.nearby(player.pos, 1)
    info = constants.make[name]
    if not all(util in nearby for util in info["nearby"]):
        return False
    if any(player.inventory[k] < v for k, v in info["uses"].items()):
        return False

    return True


class Env:
    def __init__(self, mode, args):
        assert mode in (
            "train",
            "eval",
            "yeet",
        ), "`mode` argument can either be `train`, `eval` or `yeet`"
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
                save_episode=args.game,
            )
        if mode == "yeet":
            # You are cool :)
            pass  # the vibe check
        self._obs_dim = 64
        self.env = env
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)

        # Modifiers
        self.noop = "noop" in args.agent and mode == "train"
        self.move = "move" in args.agent and mode == "train"
        self.do = "do" in args.agent and mode == "train"
        self.brain = "brain" in args.agent and mode == "train"
        self.noop_idx = self.env.action_names.index("noop")
        self.move_ids = [
            self.env.action_names.index("move_left"),
            self.env.action_names.index("move_right"),
            self.env.action_names.index("move_up"),
            self.env.action_names.index("move_down"),
        ]
        self.do_idx = self.env.action_names.index("do")
        self.table_idx = self.env.action_names.index("place_table")
        self.furnace_idx = self.env.action_names.index("place_furnace")
        self.stone_idx = self.env.action_names.index("place_stone")
        self.plant_idx = self.env.action_names.index("place_plant")

    @property
    def observation_space(self):
        return BoxSpace(0, 1, (self._obs_dim, self._obs_dim), np.float32)

    @property
    def action_space(self):
        return self.env.action_space

    def choose_action(self) -> int:
        actions = [0, 1, 2, 3, 4, 5, 6]

        if player_can_place(self.env._player, "stone"):
            actions.append(7)
        if player_can_place(self.env._player, "table"):
            actions.append(8)
        if player_can_place(self.env._player, "furnace"):
            actions.append(9)
        if player_can_place(self.env._player, "plant"):
            actions.append(10)
        if player_can_make(self.env._player, "wood_pickaxe"):
            actions.append(11)
        if player_can_make(self.env._player, "stone_pickaxe"):
            actions.append(12)
        if player_can_make(self.env._player, "iron_pickaxe"):
            actions.append(13)
        if player_can_make(self.env._player, "wood_sword"):
            actions.append(14)
        if player_can_make(self.env._player, "stone_sword"):
            actions.append(15)
        if player_can_make(self.env._player, "iron_sword"):
            actions.append(16)

        return random.choice(actions)

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

        if self.noop and action == self.noop_idx:
            reward = reward - 0.1
        if self.move and action in self.move_ids:
            reward = reward + 0.025
        if self.do and action == self.do_idx:
            reward = reward + 0.025
        if self.brain and info is not None:
            inventory = info["inventory"]
            achievements = info["achievements"]

            wood = inventory["wood"]
            if (
                wood >= 2
                and achievements["place_table"] == 0
                and action == self.table_idx
            ):
                reward += 2

            stone = inventory["stone"]
            if (
                stone >= 4
                and achievements["place_furnace"] == 0
                and action == self.furnace_idx
            ):
                reward += 2
            if (
                stone >= 1
                and achievements["place_stone"] == 0
                and action == self.stone_idx
            ):
                reward += 1

            sapling = inventory["sapling"]
            if (
                sapling >= 1
                and achievements["place_plant"] == 0
                and action == self.plant_idx
            ):
                reward += 2

        return torch.stack(list(self.state_buffer), 0).unsqueeze(0), reward, done, info


class ReplayEnv:
    def __init__(self, replaydir: str):
        self.replaydir = replaydir
        self.filenames = sorted(list(pathlib.Path(replaydir).glob(f"**/*/*.npz")))

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
) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
    class Scuff:
        def __init__(self, history_length):
            self.logdir = "logdir"
            self.history_length = history_length
            self.device = "cpu"
            self.video = False
            self.agent = "scuff"

    args = Scuff(history_length)

    env = Env("yeet", args)
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

    np.savez_compressed(cfg.logdir + "/replay.npz", **data)
