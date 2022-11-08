# Crafter Assignment starter code

This folder contains the following code:

- `train.py` A basic training loop with a random agent you can use for your own agent. Feel free to modify it at will.
- `src/crafter_wrapper.py` A wrapper over the `Crafter` environment that provides basic logging and observation preprocessing.
- `analysis/plot_stats.py` A simple script for plotting the stats of your agent.
- `analysis/plot_comp.py` A simple script for ploting the stats of all the agents as a comparison.
- `analysis/plot_duel.py` A simple script that plots the attention of the duel agent on a game.
- `analysis/plot_game.py` A simple script that plots stats of the games saved during eval.


## Instructions

Follow the installation instructions in the [Crafter repository](https://github.com/danijar/crafter). It's ideal to use some kind of virtual env, my personal favourite is `miniconda`, although installation should work with the system's python as well.

Example using venv

```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For running the Random Agent execute:

```bash
python train.py --steps 10_000 --eval-interval 2500
```

You can specify other types of agents using `--agent`. Implemented `random`, `dqn`, `ddqn`.

This will run the Random Agent for 10_000 steps and evaluate it every 2500 steps for 20 episodes. The results with be written in `logdir/random_agent/0`, where `0` indicates the run.

For executing multiple runs in parallel you could do:

```bash
for i in $(seq 1 4); do python train.py --steps 250_000 --eval-interval 25_000 & done
```

You can also run other types of agents:

- random (default)      Random agent
- dqn                   DQN Agent
- ddqn                  Double DQN Agent
- duel (modifier)       Use duel NN architecture with DQN or DDQN Agents
- ext/eext (modifier)   Use extra dataset recorded by the pro gamer himself with constant epsilon (ext) or with epsilon decay (eext)
- noop (modifier)       Use the env modifier for training which substracts 0.1 from the reward for noops
- move (modifier)       Use the env modifier for training that adds 0.025 reward if the action took is move
- do (modifier)         Same as move and noop but add 0.025 reward if action took is do
- brain (modifier)      Similar to move, noop and do. It adds extra reward if action took makes sense
                        For example if player has 2 wood and places a table it gains 2 reward
                        if the player has 4 stone and places a furnace it gains 2 reward for placing a stone it gains 1 reward
                        for placing a plant in gains 2 reward. these rewards are awarded only for the first time

Modifier means that it can be used as a decorator: `ext_duel_ddqn`

```bash
python train.py --steps 10_000 --eval-interval 2500 --agent dqn
```

### Visualization

Finally, you can visualize the _stats_ of the agent across the four runs using:

```bash
python analysis/plot_stats.py --logdir logdir/random_agent
```

You can also visualize the _stats_ of all agents using:

```console
python analysis/plot_comp.py --logdir logdir
```

For other performance metrics see the [plotting scripts](https://github.com/danijar/crafter/tree/main/analysis) in the original Crafter repo.

## TODO

1. [x] More visualization
- [x] Episodic Reward for eval
- [x] Loss plot for training
- [x] Success rate for each achievement
- [x] Distribution of actions taken with respect to time (step)
- [x] Compare the methods (reward, success rates)
- [x] Maybe try to create a plot like in the duel dqn paper? (saving the model, need to output the last layers and convert them to img)

2. [ ] More algorithms
- [x] DQN
- [x] DDQN
- [x] Dueling DQN
- [x] Maybe try to penalize noop
- [x] Explore intrinsic reward for exploring new states
- [x] Give extra reward for placing table and stuff for first time.
- [ ] Test with penalize same action multiple times in a row (or have like a diminishing return for actions) if the agent just spams space then he is bad and is a skill issue.

3. [ ] More data
- [ ] Find a dataset with prerecorded good gameplay
- [x] Record some gameplay using `python3 -m crafter.run_gui --record logdir/human_agent/0/eval`
- [x] Create a replay buffer that randomly samples from prerecorded dataset

4. [ ] More test runs to generate better plots
- [x] 3 Runs with Random
- [x] 3 Runs with DQN
- [x] 3 Runs with DDQN
- [x] 3 Runs with Duel DQN/DDQN depend on which will be better I guess
- [x] 3 Runs with extended replay buffer (from human)
- [x] 3 Runs with extended epsilon decay replay buffer (from human)
- [x] 3 Runs with noop is bad environment and all modifiers YEET

