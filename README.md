# Crafter Assignment starter code

This folder contains the following code:

- `train.py` A basic training loop with a random agent you can use for your own agent. Feel free to modify it at will.
- `src/crafter_wrapper.py` A wrapper over the `Crafter` environment that provides basic logging and observation preprocessing.
- `analysis/plot_stats.py` A simple script for plotting the stats of your agent.

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

- random (default)
- dqn
- ddqn
- duel_dqn
- duel_ddqn

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

[ ] More visualization
- [x] Episodic Reward for eval
- [x] Loss plot for training
- [x] Success rate for each achievement
- [ ] Distribution of actions taken with respect to time (step)
- [x] Compare the methods (reward, success rates)

[ ] More algorithms
- [x] DQN
- [x] DDQN
- [x] Dueling DQN
- [ ] Maybe try to penalize noop
- [ ] Explore intrinsic reward for exploring new states

[ ] More data
- [ ] Find a dataset with prerecorded good gameplay
- [ ] Record some gameplay using `python3 -m crafter.run_gui --record logdir/human_agent/0/eval`
- [ ] Create a replay buffer that randomly samples from prerecorded dataset

[ ] More test runs to generate better plots
- [x] 3 Runs with Random
- [x] 3 Runs with DQN
- [ ] 3 Runs with DDQN
- [ ] 3 Runs with Duel DQN/DDQN depend on which will be better I guess

