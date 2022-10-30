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
python train.py --steps 10_000 --eval-interval 2500 --logdir logdir/random_agent/0
```

You can specify other types of agents using `--agent`. Implemented `random`, `dqn`, `ddqn`.

This will run the Random Agent for 10_000 steps and evaluate it every 2500 steps for 20 episodes. The results with be written in `logdir/random_agent/0`, where `0` indicates the run.

For executing multiple runs in parallel you could do:

```bash
for i in $(seq 1 4); do python train.py --steps 250_000 --eval-interval 25_000 --logdir logdir/random_agent/$i & done
```

### Visualization

Finally, you can visualize the _stats_ of the agent across the four runs using:

```bash
python analysis/plot_stats.py --logdir logdir/random_agent
```

For other performance metrics see the [plotting scripts](https://github.com/danijar/crafter/tree/main/analysis) in the original Crafter repo.

## TODO

[ ] More visualization
- [x] Episodic Reward for eval
- [x] Loss plot for training
- [x] Success rate for each achievement
- [ ] Distribution of actions taken with respect to time (step)

[ ] More algorithms
- [x] DQN
- [x] DDQN
- [ ] Dueling DQN
- [ ] Maybe try to penalize noop
- [ ] Explore intrinsic reward for exploring new states

[ ] More test runs to generate better plots
- [ ] 3 Runs with Random
- [ ] 3 Runs with DQN
- [ ] 3 Runs with DDQN

