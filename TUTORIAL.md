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


# Training a Battle Network

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
The battle script will look for `.battle.data` files *recursively* in the provided dir (default=".")

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

There are the default hyperparameters. The emphasis is on speed and minimizing the number of FLOPs per inference.

### Run

```bash
(.venv) user@laptop:~$ battle -dir=first-net --data-dir=fp-data --batch-size=4096 --lr=.001 --threads=8
Using device: cpu
Saved initial network in output directory.
Initial network hash: 12608495754081121817
tensor([[0.5209, 1.0000, 0.8011, 0.7875, 1.0000],
        [0.5206, 0.0000, 0.4727, 0.4726, 0.0000],
        [0.5208, 0.0000, 0.4635, 0.4803, 0.0000],
        [0.5214, 1.0000, 0.5804, 0.3524, 1.0000],
        [0.5209, 1.0000, 0.5161, 0.6316, 1.0000]], grad_fn=<CatBackward0>)
P1 policy inference/target
tensor([[[1.4237e-01, 1.3655e-01, 1.3181e-01, 1.4931e-01, 1.4837e-01,
          1.3983e-01, 1.5175e-01, 0.0000e+00, 0.0000e+00],
         [4.8676e-03, 4.1733e-02, 1.7075e-02, 3.6469e-03, 2.2889e-04,
          9.3213e-01, 2.2889e-04, 0.0000e+00, 0.0000e+00]],

        [[1.0072e-01, 1.1029e-01, 1.2581e-01, 1.0011e-01, 1.0813e-01,
          1.1339e-01, 1.0655e-01, 1.1912e-01, 1.1589e-01],
         [2.1958e-02, 2.6352e-02, 3.2456e-02, 3.0747e-02, 7.7623e-02,
          5.6626e-02, 5.8837e-01, 1.4134e-01, 2.4399e-02]],

        [[1.0464e-01, 1.1460e-01, 1.0297e-01, 1.0395e-01, 1.1237e-01,
          1.1721e-01, 1.1067e-01, 1.1333e-01, 1.2025e-01],
         [4.7303e-04, 9.5064e-03, 4.7303e-04, 9.1138e-01, 3.7095e-02,
          4.8676e-03, 1.3413e-02, 2.2202e-02, 4.7303e-04]],

        [[1.1258e-01, 1.2462e-01, 1.1549e-01, 1.2205e-01, 1.2577e-01,
          1.3473e-01, 1.3427e-01, 1.3048e-01, 0.0000e+00],
         [1.9028e-02, 5.4321e-01, 2.4399e-02, 4.7303e-04, 4.7303e-04,
          2.5864e-02, 2.1820e-03, 3.8427e-01, 0.0000e+00]],

        [[1.0664e-01, 1.1665e-01, 1.0602e-01, 1.1697e-01, 1.1450e-01,
          9.6538e-02, 9.8539e-02, 1.1822e-01, 1.2590e-01],
         [2.9145e-03, 9.6132e-04, 6.5766e-03, 9.5064e-03, 4.7303e-04,
          2.4262e-03, 1.4745e-01, 1.8296e-02, 8.1128e-01]]],
       grad_fn=<CatBackward0>)
loss: p1:0.25835880637168884, p2:0.26018473505973816
loss: v:0.2493373304605484
```

The program print outs compare targets/predictions and display loss values. They are subject to change and won't be discussed.

We allow the training to go for 1000 steps. In this training regime (non-Network eval, low iteration, small data set) the networks seems to plateau after a few hundred steps at `lr=.001`.


## Evalutaion

The `vs` program requires the following information for both players:

* `search-budget`
* `eval`
* `bandit`
* `policy-mode`

These information can be entered with no prefix so that it applies to both players (e.g. `--search-budget=8s`)
or with the prefix `p1-`/`p2-` (e.g. `--p1-eval=fp`.)
A prefixed argument will override a non-prefixed argument.

Lets first compare the trained network with the PokeEngine eval using a think time of 1 second.
This first test does not use the networks policy inference since is using the same bandit as PokeEngine (for initial comparison's sake.)

```bash
(.venv) user@laptop:~$ vs --search-budget=1000ms --p1-eval=apple/500.battle.net --p2-eval=fp --bandit=ucb1-2.0 --policy-mode=x --threads=8 --mirror-match
score: -nan over 0 games; Elo diff: -nan
0 0 0
info: 
        0: 7, (0.50466/0.507031), (0.548622/0.550781)
        1: 8, (0.534188/0.550781), (0/0)
        2: 7, (0.508277/0.515625), (0.529301/0.53125)
        3: 7, (0/0), (0.438785/0.429688)
        4: 7, (0/0), (0.44198/0.433594)
        5: 7, (0.64514/0.648438), (0.400198/0.405924)
        6: 9, (0/0), (0.501222/0.488281)
        7: 7, (0.595607/0.592076), (0.52585/0.53125)
score: -nan over 0 games; Elo diff: -nan
```


### Args

The Pokemon-Showdown timer affords a 10 second increment, so that is probably the best search budget to use. This however makes getting (low variance) results agnonizingly slow, so we start with 1 second think time. Skill disparties seem to widen with more time.

The `ucb1` bandit is clone of FoulPlay's. It does not use policy priors since the hand crafted eval does not provide them.

### Results

```bash
score: 0.111111 over 9 games; Elo diff: -361.236
1 0 8
info: 
        0: 12, (0.389936/0.394531), (0.65818/0.664062)
        1: 141, (0.935303/0.941406), (0.54506/0.546875)
        2: 4, (0.722264/0.71875), (0.524766/0.525746)
        3: 45, (0.856075/0.854868), (0.584453/0.579086)
        4: 65, (0.662711/0.662109), (0.442586/0.433594)
        5: 146, (0.93344/0.9375), (0.909666/0.910156)
        6: 142, (0.733726/0.734375), (0.574636/0.574219)
        7: 43, (0.686821/0.685948), (0.502774/0.511719)
^CTerminated.
Terminated.
score: 0.2 over 10 games.
2 0 8
```

10 games is a *very* sample sample size but the picture is still clear: our network is outmatched with these settings. However this can easily be explained and fixed.

1. Using the exact same bandit means the network cannot use its trained policy inference

2. The network is slower than the simpler eval

Indeed, the `benchmark` tool shows that the network is abount 3x slower:

```bash
(.venv) user@laptop:~$ benchmark --eval=fp --search-budget=1000ms
1000001 ms.
253551 iterations.
(.venv) user@laptop:~$ benchmark --eval=apple/500.battle.net --search-budget=1000ms
1000008 ms.
85005 iterations.
```

The speed penalty could be greatly mitigated if it was allowed to use policy inference. Let's try that:

```bash
(.venv) user@laptop:~$ vs --search-budget=1000ms --p1-eval=apple/500.battle.net --p2-eval=fp --p1-bandit=pucb-1.0 --bandit=ucb1-2.0 --policy-mode=x --threads=8 --mirror-match
# ...
score: 0.639344 over 61 games; Elo diff: 99.4568
39 0 22
info: 
        0: 93, (0.47058/0.467969), (0.566559/0.5625)
        1: 56, (0.373886/0.371094), (0.532504/0.527344)
        2: 151, (0.888732/0.886719), (0.431848/0.421875)
        3: 121, (0.778775/0.773438), (0.595778/0.584635)
        4: 53, (0.98679/0.984375), (0.768821/0.777344)
        5: 49, (0.890679/0.886719), (0.494158/0.49486)
        6: 44, (0.964383/0.964844), (0.535601/0.542969)
        7: 202, (0.973888/0.972656), (0.520967/0.527344)
^CTerminated.
Terminated.
score: 0.639344 over 61 games.
```

With this change, the network is now 2:1 vs 'fp'.

### Conjectures



# Python Scripting

# Training a Team-Building Network

# RL

