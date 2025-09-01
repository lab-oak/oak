# First Steps

This software is intended to run on Linux. It should be able to run on Windows but that is outside the scope of these instructions. Any relatively recent distribution should work, although you may need to update your compilers since the code uses C++23 features throughout.

First, build the executables and python library using the instructions in the [README](README.md). It should be as simple as copying the commands into a terminal. Make sure Zig, CMake are installed. Most distributions of Linux only need to install the `build-essential` package for the rest of the requirements.

When running the `dev/libpkmn` script, you may notice a message about a performance increase when using an older version of Zig. For best performance, the user ideally builds that specific subversion of Zig on the target machine and uses it the build `libpkmn`. For most users it is enough that they use the [Zig 11.0 release](ziglang.org/download/#release-0.11.0) binary to build. However, even the worse case (current Zig) is still plenty fast enough.

## benchmark

This program is not part of the usual workflow. It simply runs a search on a fixed battle for 2^n iterations. 'n'=20 by default (1,048,576 iterations) but it can be set as the first argument to the program, e.g.

```
./build/benchmark 10
```

The search completes in ~22 seconds on very modest hardware (AMD Ryzen 5 5500U). The above command and all others assume the user is currently in the oak/ directory.

# Programs/Workflow

The programs are flexible in how the can be used and they accept many different keyword arguments. This is so that the user does not need to program or modify the source in order to experiment.

If the user wants to make changes to the source, it is recommended they read [this](include/README.md) first.

## chall

### About

This is the 'analysis engine' and the most interactive part of Oak. The program accepts two optional args on startup: the first for the path to the saved network parameters and the second for the specific search algorithm. By default the network path is "mc" which indicates that Monte-Carlo evaluation should be used instead. The default selection algorithm is [Exp3](https://en.wikipedia.org/wiki/Multi-armed_bandit#Exp3) with a 'gamma' value of 0.03.

```

Monte Carlo evaluation is simply taking a game position and playing random moves until the game is concluded. The score that each player receives is used as the value estimation for the original position. This evaluation is incredibly simple and sufficient for the search to converge to a Nash equilibrium. It also outperforms all hand crafted evals I have tried.

On the other hand it can be slow (it can take hundreds of turns for a 6v6 battle to terminate with random moves and especially switches.) It is also clearly very weak since it only offers values of 0.0, 0.5, 1.0 from a random process.

```

The considerations above make the strength of Monte-Carlo search highly context dependent. In positions with low number of Pokemon and an absence of recovery moves this eval is competent. Conversely, is is rather hopeless in a full 6v6. The real question: What positions is Monte-Carlo strong/correct with a reasonable search time?

With the disclaimers out of the way, let's try using the program

### Reading the Program Output

```
./build/chall
```

the program outputs

```bash
network path: mc
bandit algorithm: exp3-0.03
Enter battle string:
```

A battle string is a compact text representation of a battle position. It takes the form

`[Side] | [Side]`

where a Side takes the form

`[Pokemon]; [Pokemon] ...`

A Pokemon must begin with a species. The remaining information (moves, status, etc) can appear in any order. The species and moves do not have to be spelled out entirely. A partial spelling will be accepted if it matches the initial segment of a unique move/species. The format is case insensitive. This means that

`Snorlax BodySlam Reflect Rest Earthquake | slowb psychic rest Amnes`

is valid input while

`Snorlax BodySlam Reflect Rest Earthquake | slow psy rest Amnes`

is not, since 'slow' matches both Slowpoke and Slowbro, and 'psy' matches multiple moves.

Let's try this position out, but first we can make it a bit easier for the program. It is clear that Snolax's moves Reflect and Earthquake are not useful. So let's just remove them from the battle string.

```bash
Battle:
Snorlax: 100% (523/523) BodySlam:24 Rest:16 None:0 None:0 
-
Slowbro: 100% (393/393) Psychic:16 Amnesia:32 Blizzard:8 Rest:16 

P1 choices:
0: BodySlam 1: Rest 
P2 choices:
0: Psychic 1: Amnesia 2: Blizzard 3: Rest 
Starting search. Suspend (Ctrl + Z) to stop.
```

The search will continue until paused (Ctrl + Z) or the program is terminated (Ctrl + C).

Memory usage will increase as the search tree gets larger. Users with low memory (8Gb or less) should watch it to get a sense of how quickly it rises. The search tree is cleared after the root battle is updated but the programs alloted memory does not typically decrease.

If we pause the program we get a summary of the search results.

```
Iterations: 1058462, Time: 6.572 sec
Value: 0.337

P1
BodySlam  Rest      
0.985     0.015     
1.000     0.000     
P2
Psychic   Amnesia   Blizzard  Rest      
0.447     0.293     0.251     0.008     
0.000     0.000     1.000     0.000     
ge
Matrix:
         Psychic  Amnesia  Blizzar  Rest     
BodySla  0.335    0.349    0.335    0.383    
Rest     0.133    0.182    0.161    0.215    
Input: P1 index (P2 index); Negative index = sample.
```

The 'value' is the expected value for player 1 at the root position we've entered. Player 1 wins are given a value of 1.0, losses 0.0, and draws 0.5. This suggests that the Slowbro is favored.

The bottom portion is the 'empirical matrix': During each MCTS iteration, both players use their bandit algorithms to select an action. The pairs of actions define a matrix and each iteration will update a single entry matrix entry. This entry stores the average value which depends on chance and subsequent player actions further in the tree.

Lastly we have the search policies. The empirical policies are just the number of times the bandit algorithm chose that action at the root node, divided by the number of iterations. As a result it almost always has non-zero probability for each action, even terrible ones. The 'Nash' policy is more refined. It is produces by solving Nash equilibrium to the empirical matrix:
    
```
The following might be helpful for understanding the 'Nash' aspect if the reader is familiar with vanilla MCTS setups.

In the famous RL algorithm AlphaZero, the search uses a UCB-backed MCTS. During self-play training, the algorithm selects moves in proportion to the empirical policy. This is to balance playing good moves with exploration. However during the testing phase (playing vs Stockfish) the moves with the highest average value were selected.

The policy given by solving the emprical value matrix is the '2D' analog of the greedy policy used in the testing phase of AlphaZero.

```

### Making Moves

Below the search summary we see

```
Input: p1 index (p2 index); Negative index = sample.
```

Depending on our input we can either commit actions and advance the battle position or we can resume the search. Any input that can't be parsed as a a lone integer or pair of integers will resume the search. Simply hitting enter with no text entered will suffice.

Let's allow the search to continue for a while.

```
Iterations: 52584005, Time: 386.654 sec
Value: 0.419

P1
BodySlam  Rest      
0.985     0.015     
1.000     0.000     
P2
Psychic   Amnesia   Blizzard  Rest      
0.367     0.354     0.272     0.008     
0.000     0.000     1.000     0.000     

Matrix:
         Psychic  Amnesia  Blizzar  Rest     
BodySla  0.418    0.432    0.415    0.497    
Rest     0.155    0.189    0.177    0.274  
```

This second output includes the information from the first search period, e.g. the first 1,058,462 iterations are counted among the second 52,584,005 iterations, etc.

The nash strategies have not changed but the expected score for Snorlax has risen somewhat. The truth is that calculating the actual expected value for this position is difficult for MCTS since both pokemon know rest. Without critical hits the Slowbro should be able to stall out the Body Slams but a crit + speed tie loss at the right turn should result in a Snorlax victory.

The folling should explain how selecting moves works. 'Sampled' means the index is sampled from that player's nash policy.

| Input | P1 Move | P2 Move | 
| ----------- | ----------- | ----------- |
| 0 | Body Slam | Sampled |
| 1 | Rest | Sampled | 
| -1 0 | Sampled | Psychic |
| -1 3 | Sampled | Rest |
| -1 -1 | Sampled | Sampled |

When the battle is concluded the score is printed and the program terminates.

## Pyoak

This is not a program but rather a library of functions that can be used in a Python script.

The functions are exposed to python in [this](include/pyoak/py) module. The functions are one-to-one with the underlying C++ implementations [here](src/pyoak.cc).

The data structures `Battle::Frame` (raw training data for battling), `Battle::Frame` (where battle bytes are converted into tensors), and `BuildTrajectory` are mirrored in Python as well. The python versions are 'batched' and the data is stored as Numpy arrays so they work with `Pytorch` without any further conversion.

This library allows users to view and manipulate training data without any C++ programming knowledge. If you can follow a Medium article you can train a RBY neural network.

### tutorial.py

This script was writting for this tutorial. Run it with the various args to view the results of the rest of this guide.

## generate

This program uses self-play to play-out battles. Each turn will have value and policy targets associated with it. The program accepts many keyword arguments and it would take too long to explain them here. For this tutorial, will run the program with settings that allow millions of battle and team generation samples to be created within an hour on a consumer machine.

```
./build/generate --threads=8 --max-pokemon=1 --modify-team-prob=1 --pokemon-delete-prob=1
```

### Arguments Explained

* `--threads=8`

My laptop has 6 cores with 2 threads per, but I use only 2/3 of them so that I can run this program in the background.

* `--max-pokemon=1`

This is what allows quick generation. All the battles will be 1v1's which conclude much faster than a full 6v6. Also, the default evaluation is Monte-Carlo, so the searches run much faster as well since the rollouts are much shorter too.

Besides the practical reasons for this setting, it may be possible to train battle and team builder networks for the full 6v6 by starting on 1v1's and increasing the mon count. The rest of the turorial will show how this can be done by simply running the vanilla programs with different arguments

* `--modify-team-prob=1`

The program accepts an argument `teams` for the path to a line separated list of teams in the battle string format described in the #chall section. If not provided, the program will use the hard-coded OU sample teams [here](include/data/teams.h). The teams are the 'base'.

A value of 1 means the program will attempt to delete information from the team and fill it back it with the team generation network. This is done at the start of each battle for both players.

* `--pokemon-delete-prob=1`

The probabilty that the entire set (species and moves) will be deleted and filled in. This and the previous two args mean that the build network will generate each single-pokemon team from scratch.


### Output

The program will automatically create a directory named after the date-time of creation.

Statistics about the data generation speed will display periodically. The 'keep-node' ratio is the percentage of search trees that were retained after a battle update.

After some time we end the program with `Ctrl + C`. If we inspect the contents of this run's work dir

```
~/oak$ ls 2025-08-15-20:55:37
0.battle  1.build   30.build  41.build  52.build  63.build  74.build  85.build  97.build
0.build   20.build  31.build  42.build  53.build  64.build  75.build  86.build  98.build
10.build  21.build  32.build  43.build  54.build  65.build  76.build  87.build  9.build
11.build  22.build  33.build  44.build  55.build  66.build  77.build  88.build  args
12.build  23.build  34.build  45.build  56.build  67.build  78.build  89.build  build-network
13.build  24.build  35.build  46.build  57.build  68.build  79.build  8.build
14.build  25.build  36.build  47.build  58.build  69.build  7.battle  90.build
15.build  26.build  37.build  48.build  59.build  6.battle  7.build   91.build
16.build  27.build  38.build  49.build  5.battle  6.build   80.build  92.build
17.build  28.build  39.build  4.battle  5.build   70.build  81.build  93.build
18.build  29.build  3.battle  4.build   60.build  71.build  82.build  94.build
19.build  2.battle  3.build   50.build  61.build  72.build  83.build  95.build
1.battle  2.build   40.build  51.build  62.build  73.build  84.build  96.build
```

we can see a large number of '.build' files. The ratio of build to battle files is skewed compared to full 6v6 self-play. Each build file contains ~128K teams and the battle files are 8mb of raw battle/update/stat information by default. 

```
~/oak$ cat 2025-08-15-20:55:37/args
--threads=8
--seed=4127720828
--max-frames=1073741824
--buffer-size=8
--debug-print=true
--print-interval=30
--search-time=4096
--bandit-name=exp3-0.03
--battle-network-path=mc
--policy-mode=n
--policy-temp=1
--policy-min-prob=0
--policy-nash-weight=0.5
--keep-node=true
--skip-game-prob=0
--max-pokemon=1
--teams=
--build-network-path=
--modify-team-prob=1
--pokemon-delete-prob=1
--move-delete-prob=0
--t1-search-time=4096
--t1-bandit-name=exp3-0.03
--t1-battle-network-path=mc
```

reports the args used. The build network path is empty/was not provided so a new network was initialized and saved in the work dir as `build-network`.

### Inspecting the Data with Python

```bash
~/oak$ python3 py/tutorial.py read-build-trajectories
```

This option will print the first 10 build trajectories from the first build file it finds in the `oak` directory.

```
Sample 0:
[(('Onix', 'None'), 0.006469825282692909), (('Onix', 'Substitute'), 0.04887464642524719), (('Onix', 'Bind'), 0.05099565163254738), (('Onix', 'Explosion'), 0.05064469203352928), (('Onix', 'SkullBash'), 0.05670252442359924), (('Onix', 'None'), 1.0)]
```

The network is freshly initialized so its predictions for species/moves are basically uniformly random. Any OU legal pokemon is possible and any of its legal moves is possible. This project does not consider the few illegal moveset combinations that exist in RBY; They may be picked but they are not competitively relevant.

The actions for the network are all possible '(OU Pokemon, Move it Learns by lvl 100)' pairs. If the move is `None` it means that action was picking the species to begin with. The final such pair is the selection of the teams lead.

```
(pokemon 1, None), (pokemon 2, None), (pokemon 1, move 1)
```

The probability the net produced for selecting the move is stored. This value is necessary for Policy Gradient learning algorithms. The build network is simply rolled out - not used in any search/planning capacity. This means we cannot produce entire policy targets like the battle net does.

### Training a Build Network

TODO is a simple script for training using [PPO](https://en.wikipedia.org/wiki/Proximal_policy_optimization). It is an on-policy algorithm, which means it works best if trained using data that it produced itself (rather than say a previous version of the network).

// Run the program

// List karg for using the network in another data generation run

// Visually inspect build rollouts of this new network (hopefull observe no onix lol)

### Training a Battle Network

```
~/oak$ python3 py/train.py
usage: train.py [-h] [--threads THREADS] [--steps STEPS] [--checkpoint CHECKPOINT] [--lr LR]
                [--write-prob WRITE_PROB] [--w-nash W_NASH] [--w-empirical W_EMPIRICAL]
                [--w-score W_SCORE] [--w-nash-p W_NASH_P] [--no-policy-loss] [--seed SEED]
                [--device DEVICE] [--print-window PRINT_WINDOW]
                net_dir battle_dir batch_size
train.py: error: the following arguments are required: net_dir, battle_dir, batch_size
```

A few kargs deserve some explanation

* `--write-prob=`

* `--w-score`/`--w-nash`/`--w-empirical`

* `--threads`

Of the required positional args

* `net_dir`

Output directory to be created for network weights

* `battle_dir`

Directory to recursively scan for battle files. Simply `.` will scan the current working (oak) directory and find all data from all runs. If you want to restrict to to a specific one that just provided that path.

* `batch_size`

Batch size for the optimizer.

```
~/oak$ python3 py/train.py --steps=2000 --w-empirical=1 --
```

## Iteration

### Second Data Generation

