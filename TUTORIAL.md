# Installation and Activation

This tutorial will assume that the user has installed the package in a virtual enviroment.

When the venv is active, several python scripts will be visible to terminal

```bash
user@laptop:~$ battle
Command 'battle' not found, did you mean:
  command 'bottle' from snap bottle (0.0.4)
See 'snap info <snapname>' for additional versions.
user@laptop:~$ source .venv/bin/activate
(.venv) user@laptop:~$ battle
usage: battle [-h] [--network-path NETWORK_PATH] [--dir DIR] [--data-dir DATA_DIR] [--in-place]
              [--steps STEPS] [--device DEVICE] [--threads THREADS] --batch-size BATCH_SIZE --lr LR
              [--lr-decay LR_DECAY] [--lr-decay-start LR_DECAY_START]
              [--lr-decay-interval LR_DECAY_INTERVAL] [--data-window DATA_WINDOW] [--min-files MIN_FILES]
              [--sleep SLEEP] [--checkpoint CHECKPOINT] [--delete-window DELETE_WINDOW] [--seed SEED]
              [--max-battle-length MAX_BATTLE_LENGTH] [--min-iterations MIN_ITERATIONS]
              [--no-clamp-parameters] [--value-nash-weight VALUE_NASH_WEIGHT]
              [--value-empirical-weight VALUE_EMPIRICAL_WEIGHT] [--value-score-weight VALUE_SCORE_WEIGHT]
              [--p-nash-weight P_NASH_WEIGHT] [--no-value-loss] [--no-policy-loss]
              [--policy-loss-weight POLICY_LOSS_WEIGHT] [--no-apply-symmetries]
              [--pokemon-hidden-dim POKEMON_HIDDEN_DIM] [--active-hidden-dim ACTIVE_HIDDEN_DIM]
              [--pokemon-out-dim POKEMON_OUT_DIM] [--active-out-dim ACTIVE_OUT_DIM]
              [--hidden-dim HIDDEN_DIM] [--value-hidden-dim VALUE_HIDDEN_DIM]
              [--policy-hidden-dim POLICY_HIDDEN_DIM] [--print-window PRINT_WINDOW]
battle: error: the following arguments are required: --batch-size, --lr
(.venv) user@laptop:~$ 
```

The package is available on PyPI here and can be installed with

```
TODO make venv, activate, install
```

Alternatively, the wheel can also be downloaded from the Github releases.

```
```


# Beating PokeEval

One of the successes of this project is the ease and consistency with which a user can train a network that is stronger than FoulPlay's hand crafted eval. In this section, we will duplicate this result on a low-end consumer laptop.

The general plan:

* Fast self-play using `generate --eval=fp`

* Train battle script from scratch using `battle`

* Compare strength using `vs`

First let's use the benchmark tool to get an idea of how fast data generation is 

```bash
(.venv) user@laptop:~$ benchmark --eval=fp --bandit=ucb-1.0 --search-budget=1024
3936 ms.
1024 iterations.
(.venv) user@laptop:~$ benchmark --eval=fp --bandit=ucb-1.0 --search-budget=4096
16779 ms.
4096 iterations.
(.venv) user@laptop:~$ python
Python 3.13.3 (main, Jan  8 2026, 12:03:54) [GCC 14.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> fast = 2653 # microseconds for fast search
>>> slow = 6739 # full search and valid sample
>>> expected = (7/8) * fast + (1/8) * slow
>>> spus = 1 / expected # step per microsec
>>> s = spus * 10**6
>>> b = s / 80 # average battle length is 80, so battles per second
>>> x = 8 * b # times number of threads
>>> y = 10**6 / x # seconds to 1M battles
>>> z = y / 3600 # hours
>>> print(z)
8.788194444444445
>>> 
```

The point of this calculation is to estimate how long it will take to generate 1M battles aka samples for the network' value output. I have succeeded with only a 250K for reference. This is likely around the lower bound for network superiority in test time regimes, and performance improves with more battles.

## Data generation

Let's pick some arguments for the `generate` call  with experience in mind

* `--search-budget=4096`

2^12 is a reasonable iteration count for a few reasons. AlphaZero used 1000 iterations, and the branching factor for a simultaneous move game is the product of the number of actions for either player. Its totally possible RBY has a higher average branching factor and that's not even considering RNG. Therefore we probably at least as many iterations as altenative move cofigs.

* `--teams=teams`

Unfortunately I can't share this 

* `--fast-search-prob=`

This is an innovation of Efficient MuZero. It reduces cost of data gen while balancing value and policy learning.

* `--fast-search-budget=`

* `--bandit=`

At the time of this writing the strongest

UCB is very aggresive which suits the low iteration count. This is also true for argmax move selection as well. It should be noted that Nash move selection is very sensitive to variance in any of the matrix entries - all move pairs must be decently explored for it to be competitive with argmax

* `--threads=8`

The max for my system. By default it does max - 1.

* `--dir=fp-data`

The name of the dir where all work will be saved. By default this is a datatime string. It is advised to use short code names.

```bash
(.venv) user@laptop:~/oak$ generate --search-budget=1024 --bandit=ucb-1.0 --policy-mode=x --eval=fp --dir=fp-data
Created directory fp-data
279.767 battle frames/sec.
keep node ratio: 0
progress: 0.000781659%
Game Lengths: 27 33 32 20 2 2 1 
```

You should see something like this. A line confirming the work directory was created and some periodic 

After some time has passed we enter `Ctrl + C` to send a SIGINT signal. This triggers a save

All the Oak programs save their arguments to disk

```bash
(.venv) user@laptop:~/oak$ ls fp-data
0.battle.data  2.battle.data  4.battle.data  6.battle.data  matchup-matrix
1.battle.data  3.battle.data  5.battle.data  args
(.venv) user@laptop:~/oak$ head -10 fp-data/args
   --team-modify-prob : 0
--pokemon-delete-prob : 0
   --move-delete-prob : 0
 --build-network-path : 
        --max-pokemon : 6
      --search-budget : 1024
             --bandit : ucb-1.0
         --matrix-ucb : 
               --eval : fp
```


## Training

Let's start training by running a quick check.

```bash
(.venv) user@laptop:~$ tutorial battle-frame-stats fp-data
Total battle frames: 19373207
Average battle length: 80.1695282078021
(.venv) user@laptop:~$ python
Python 3.13.3 (main, Jan  8 2026, 12:03:54) [GCC 14.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 19373207 / 80 # number of games
242165.0875
>>> 
```

The keyword args for the battle script are lean enough that we can discuss each of them below. All Oak scripts will list their kwargs if the help flag is provided, e.g.

```bash
battle --help
```


* `--data-dir=fp-data`

* `--batch-size=4096`
Pokemon scores have higher variance because of the stochastic nature of the game.

* `--lr=.001`
A proven constant and the default for Adam (the optimization we use)

* `--threads=8`
Unlike `generate`, this script uses only one thread by default. This kwarg limits the max number of threads that Torch/CUDA can use and the number of data reading threads.

The following arguements are default and were not explicitly entered, but deserve mention anyway.

* `--value_nash_weight=0.0`
* `--value_empirical_weight=0.0`
* `--value_score_weight=1.0`
The final value target is a weighted sum of 3 different value estimates.
The empirical value is just the average leaf value that is back progpogated to the root. The nash value is more complicated, but it is the value corresponding to a Nash equilibrium on the root node's "empirical matrix".

Most RL setups use only the score as we have. Additionally, using the PokeEngine eval we used for self play is not intended to be a value estimator. This means that the empirical and Nash values are less meaningful than if we used Monte-Carlo or a Network.

* `--p_nash_weight=0.0`
Nash solving isnt good when UCB is used.

* `--pokemon_hidden_dim=128`
* `--active_hidden_dim=128`
* `--pokemon_out_dim=59`
* `--active_out_dim=83`
* `--hidden_dim=64`
* `--value_hidden_dim=32`
* `--policy_hidden_dim=64`





# Python

# Battle Network

# RL

