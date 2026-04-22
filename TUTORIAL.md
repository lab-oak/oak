# Installation and Activation

Oak is available on the python package index under [oak-lab](https://pypi.org/project/oak-lab/).

It is recommended that you install Oak in a virtual environment:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install oak-lab
```

If the installation fails, it is likely that you are using either an unsupported Python version/OS/CPU architecture. Currently, **Oak is only available for Python versions 3.10 - 3.14 on the Linux operating system with x86-64 architecture.** Windows users can use [WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).

The installation can quickly be checked with the `benchmark` program, which will run ~1M iterations of pure MCTS on turn 1 of a 6v6 game

```bash
(.venv) $ benchmark
12811ms.
1048576 iterations.
```

While a Oak-installed virtual environment is active, the following binaries will be available from command line:

* `benchmark`
* `oak-search-test`
* `generate`
* `chall`
* `vs`

 and the following Python scripts:

* `lab`
* `battle`
* `build`
* `evo`

The usage of all these programs will be covered in this tutorial.

Oak is also a traditional Python library:

```bash
(.venv) $ python
Python 3.13.3 (main, Jan  8 2026, 12:03:54) [GCC 14.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import oak
>>> input = oak.parse_battle("snorlax bodyslam rest | starmie psychic thunderwave recover")
>>> agent = oak.Agent()
>>> agent.budget = "8s"
>>> agent.bandit = "ucb-1.0"
>>> heap = oak.Heap()
>>> output = oak.search(input, heap, agent)
>>> print(oak.format(input, output))
Iterations: 847615, Time: 8000.02 sec
Value: 0.448

P1
BodySlam  Rest      
0.989     0.011     
1.000     0.000     
P2
Psychic   ThunderWave  Recover   
0.028     0.970     0.001     
0.000     1.000     0.000     

Matrix:
         Psychic  Thunder  Recover  
BodySla  0.602    0.443    0.767    
Rest     0.395    0.398    0.485    

Visits:
         Psychic  Thunder  Recover  
BodySla  23811    813826   943      
Rest     329      8673     33       
```

# Training a Battle Network

The example above was chosen to illustrate that Oak is capable of replacing the search used in IS-MCTS projects like [Foul Play](https://github.com/pmariglia/foul-play). But the evaluation used in the example is Monte Carlo, not a (much stronger) trained Oak battle network. Let's begin with data generation and network training.

The general plan:

* Fast self-play using `generate`

* Train value/policy network using `battle`

* Compare strength (relative to FoulPlay and Monte-Carlo) using `vs`

## Data generation

The `generate` program accepts many keyword arguments but only a few are required. Those that are reflect basic considerations that we will discuss now:

* `--eval`

This is value estimator that the search will use in self-play games. To start with, we only have PokeEngine evaluation ("fp") and Monte-Carlo ("mc", default).

Despite being a simple hand crafted score function, "fp" is much stronger and faster than Monte-Carlo.

```
(.venv) $ vs --budget=4096 --bandit=ucb-1.0 --policy-mode=x --p1-eval=fp --p2-eval=mc
...
W D L:
186 1 31
```

Above we've used the `vs` command with sensible arguments to compare the strength of these two evals, and we can see "fp" scored 186 wins to "mc"s 31.

Therefore we will use the "fp" eval function our first self-play data generation run.

* `--budget=4096`

2^12 is a reasonable iteration count for a few reasons. AlphaZero used 1000 ~ 2^10 iterations, and the branching factor for a simultaneous move game is the product of the number of actions for either player. Its totally possible RBY has a higher average branching factor after including RNG. Therefore we probably should use at least as many iterations as altenative move cofigs.

```bash
(.venv) $ benchmark --eval=fp --budget=4096
3548µs.
4096 iterations.
```

On my machine this gives us 3.5 milliseconds per step.

* `--bandit=`

There are 5 bandit algorithms available:

* `ucb`
* `pucb`
* `ucb1`
* `exp3`
* `pexp3`

Each of these has a float parameter that comes afterwards separated by a '-', e.g. `ucb-1.0`. For the 'ucb` variants this is the exploration weight "c" and for 'exp3' variants it is the update weight "gamma". The exp3 variants have a second optional parameter which is the weight of the uniform policy noise in the forecast.

Currently all evidence points to ucb being the strongest variant, despiate exp3's [theoretical guarantees](TODO link paper?). It is probably also better suited towards low iteration searches.

* `--policy-mode=x`

The search will produce multiple strategies or policies for either player. These are:

* `e` emprical
* `x` argmax
* `n` nash
* `p` prior
* `u` uniform

All of these characters are valid arguments. Additionally, weighted combinations may be passed e.g. `x0.9e0.1`.

* `--teams=teams`

Any time teams are required Oak will default to the 17 or so Smogon sample teams. More teams is certainly beneficial however. The programs expect a simple plaintext format that can be explained hopefully with just an example (2 lines/teams):

```
jynx blizzard lovelykiss psychic rest; chansey counter icebeam softboiled thunderbolt; jolteon doublekick rest thunderbolt thunderwave; snorlax bodyslam icebeam reflect rest; starmie blizzard psychic recover thunderwave; tauros blizzard bodyslam earthquake hyperbeam
gengar confuseray explosion hypnosis thunderbolt; chansey icebeam softboiled thunderbolt thunderwave; cloyster blizzard clamp explosion rest; exeggutor explosion psychic rest sleeppowder; golem earthquake explosion rest rockslide; slowbro amnesia rest surf thunderwave
```

* `--fast-search-prob=`

This is an innovation of Efficient MuZero. It reduces cost of data gen while balancing value and policy learning.

* `--fast-budget=`


* `--threads=8`

The max for my system. By default it does max - 1.

* `--dir=fp-data`

The name of the dir where all work will be saved. By default this is a datetime string. It is advised to use short code names.

```bash
(.venv) user@laptop:~/oak$ generate --budget=1024 --bandit=ucb-1.0 --policy-mode=x --eval=fp --dir=fp-data
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
      --budget : 1024
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

* `budget`
* `eval`
* `bandit`
* `policy-mode`

These information can be entered with no prefix so that it applies to both players (e.g. `--budget=8s`)
or with the prefix `p1-`/`p2-` (e.g. `--p1-eval=fp`.)
A prefixed argument will override a non-prefixed argument.

Lets first compare the trained network with the PokeEngine eval using a think time of 1 second.
This first test does not use the networks policy inference since is using the same bandit as PokeEngine (for initial comparison's sake.)

```bash
(.venv) user@laptop:~$ vs --budget=1000ms --p1-eval=apple/500.battle.net --p2-eval=fp --bandit=ucb1-2.0 --policy-mode=x --threads=8 --mirror-match
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

The data display is of the format

```
  {thread}: {update}, ({p1_output.empirical_value}/{p1_output.nash_value}), ({p1_output.empirical_value}/{p1_output.nash_value})
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
^Cscore: 0.2 over 10 games.
2 0 8
```

10 games is a *very* sample sample size but the picture is still clear: our network is outmatched with these settings. However this can easily be explained and fixed.

1. Using the exact same bandit means the network cannot use its trained policy inference

2. The network is slower than the simpler eval

Indeed, the `benchmark` tool shows that the network is abount 3x slower:

```bash
(.venv) user@laptop:~$ benchmark --eval=fp --budget=1000ms
1000001 ms.
253551 iterations.
(.venv) user@laptop:~$ benchmark --eval=apple/500.battle.net --budget=1000ms
1000008 ms.
85005 iterations.
```

The speed penalty could be greatly mitigated if it was allowed to use policy inference. Let's try that:

```bash
(.venv) user@laptop:~$ vs --budget=1000ms --p1-eval=apple/500.battle.net --p2-eval=fp --p1-bandit=pucb-1.0 --bandit=ucb1-2.0 --policy-mode=x --threads=8 --mirror-match
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
^Cscore: 0.639344 over 61 games.
```

With this change, the network is now 2:1 vs 'fp'.

### Conjectures



# Python Scripting

# Training a Team-Building Network

# RL

