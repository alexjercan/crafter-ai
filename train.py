import os
import torch
import argparse
import pickle

from tqdm.auto import tqdm
from pathlib import Path
from src.crafter_wrapper import Env
from src.utils import *
from torch import Tensor


class DQN_Conv(nn.Module):
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

    def forward(self, x):
        x = self.blocks(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


class DQNAgent:
    def __init__(self, Q: nn.Module, device: str = "cpu") -> None:
        self.Q = Q
        self.device = device

    @torch.no_grad()
    def act(self, s: Tensor):
        s = s[None].to(self.device)
        return self.Q(s).argmax(dim=1).item()


def dqn_agent(opt, target_function):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = opt.batch_size
    gamma = opt.gamma
    replay_buffer_size = opt.replay_buffer_size
    learning_starts = opt.learning_starts
    learning_freq = opt.learning_freq
    target_update_freq = opt.target_update_freq
    device = opt.device

    env = Env("train", opt)
    eval_env = Env("eval", opt)

    input_shape = env.observation_space.shape
    input_arg = env.window * input_shape[0] * input_shape[1]
    num_actions = env.action_space.n

    Q = DQN_Conv(env.window, num_actions, input_shape)
    target_Q = DQN_Conv(env.window, num_actions, input_shape)

    Q.apply(init_weights)
    target_Q.apply(init_weights)

    optimizer = optim.RMSprop(Q.parameters())
    criterion = nn.HuberLoss()
    replay_buffer = ReplayBuffer(replay_buffer_size)
    eps_scheduler = iter(linear_eps_generator())

    best_avg_return = 0

    agent = DQNAgent(Q, device)

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    pbar = tqdm(total=opt.steps, position=0, leave=True)
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            s, done = env.reset(), False
            episode_reward = 0

        if step_cnt > learning_starts:
            eps = next(eps_scheduler)
            a = select_epsilon_greedy_action(env, Q, s, eps)
        else:
            a = env.action_space.sample()

        s_next, r, done, _ = env.step(a)

        episode_reward += r

        replay_buffer.store(s, a, r, s_next, done)

        s = s_next

        if step_cnt > learning_starts and step_cnt % learning_freq == 0:
            (
                s_batch,
                a_batch,
                r_batch,
                s_next_batch,
                done_batch,
            ) = replay_buffer.sample(batch_size)

            s_batch = s_batch.float().to(device)
            a_batch = a_batch.long().to(device)
            r_batch = r_batch.float().to(device)
            s_next_batch = s_next_batch.float().to(device)
            done_batch = done_batch.long().to(device)

            Q_values = Q(s_batch).gather(1, a_batch.unsqueeze(1)).view(-1)

            target_Q_values = target_function(
                Q, target_Q, r_batch, s_next_batch, done_batch, gamma
            )

            loss = criterion(target_Q_values, Q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step_cnt > learning_starts and step_cnt % target_update_freq == 0:
            target_Q.load_state_dict(Q.state_dict())

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            avg_return = eval(agent, eval_env, step_cnt, opt)

            if avg_return > best_avg_return:
                best_avg_return = avg_return

                torch.save(
                    Q.state_dict(),
                    os.path.join(opt.logdir, f"model_{step_cnt}.pth"),
                )

        pbar.set_description(
            f"[Episode {ep_cnt}]: Current reward {episode_reward:.04f}"
        )
        pbar.update(1)
        step_cnt += 1


class RandomAgent:
    """An example Random Agent"""

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """Since this is a random agent the observation is not used."""
        return self.policy.sample().item()


def random_agent(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Env("train", opt)
    eval_env = Env("eval", opt)
    agent = RandomAgent(env.action_space.n)

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False

        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        step_cnt += 1

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)


def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    tqdm.write(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)

    return avg_return


def eval(agent, env, crt_step, opt):
    """Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    return _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},64,64),"
        + "with values between 0 and 1."
    )


def main(opt):
    if opt.agent == "random":
        return random_agent(opt)
    if opt.agent == "dqn":
        return dqn_agent(opt, target_function=dqn_target)
    if opt.agent == "ddqn":
        return dqn_agent(opt, target_function=ddqn_target)

    raise NotImplementedError(f"Agent {opt.agent} not implemented yet.")


def get_options():
    """Configures a parser. Extend this with all the best performing hyperparameters of
    your agent as defaults.

    For devel purposes feel free to change the number of training steps and
    the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10_000,
        dest="eval_interval",
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount Factor",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=10_000,
        help="Replay buffer size",
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=1000,
        help="Number of steps to use random sampling for replay buffer",
    )
    parser.add_argument(
        "--learning_freq",
        type=int,
        default=4,
        help="Number of steps to wait in between learning steps",
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=100,
        help="Number of steps to switch the target Q network and the Q network",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        help="The name of the agent that you want to run (random, dqn, ddqn)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Save video of eval process",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = get_options()

    main(get_options())
