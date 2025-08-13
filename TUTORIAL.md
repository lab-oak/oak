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

### about

This is the 'analysis engine' and the most interactive part of Oak. The program accepts two optional args on startup: the first for the path to the saved network parameters and the second for the specific search algorithm. By default the network path is "mc" which indicates that Monte-Carlo evaluation should be used instead. The default selection algorithm is [Exp3](https://en.wikipedia.org/wiki/Multi-armed_bandit#Exp3) with a 'gamma' value of 0.03.

```

Monte Carlo evaluation is simply taking a game position and playing random moves until the game is concluded. The score that each player receives is used as the value estimation for the original position. This evaluation is incredibly simple and sufficient for the search to converge to a Nash equilibrium. It also outperforms all hand crafted evals I have tried.

On the other hand it can be slow (it can take hundreds of turns for a 6v6 battle to terminate with random moves and especially switches.) It is also clearly very weak since it only offers values of 0.0, 0.5, 1.0 from a random process.

```

The considerations above make the strength of Monte-Carlo search highly context dependent. In positions with low number of Pokemon and an absence of recovery moves this eval is competent. Conversely, is is rather hopeless in a full 6v6. The real question: What positions is Monte-Carlo strong/correct with a reasonable search time?

With the disclaimers out of the way, let's try using the program

### Reading the Program Output

```bash

./build/chall

```

the program outputs

```bash

network path: mc
bandit algorithm: exp3-0.03
Enter battle string:

```

Battle strings are in the form

`[Side] | [Side]`

where a Side takes the form

`[Pokemon]; [Pokemon] ...`

A Pokemon must begin with a species. The remaining information (moves, status, etc) can appear in any order. The species and moves do not have to be spelled out entirely. A partial spelling will be accepted if it matches the initial segment of a unique move/species. The format is case insensitive. This means that

`Snorlax BodySlam Reflect Rest Earthquake | slowb psychic rest Amnes`

is valid input while

`Snorlax BodySlam Reflect Rest Earthquake | slow psy rest Amnes`

is not, since 'slow' matches both Slowpoke and Slowbro, and 'psy' matches multiple moves.

If we input the correct string into the program we get a print of the parsed battle position

```bash

Battle:

Snorlax: 100% (523/523) BodySlam:24 Reflect:32 Rest:16 Earthquake:16

Slowbro: 100% (393/393) Psychic:16 Rest:16 None:0 None:0

P1 choices:

0: BodySlam 1: Reflect 2: Rest 3: Earthquake

P2 choices:

0: Psychic 1: Rest

Starting search. Suspend (Ctrl + Z) to stop.

```

The search will continue until paused (Ctrl + Z) or the program is terminated (Ctrl + C).

Note that the memory usage will almost always rise slowly as the search continues. Uses with low memory (8Gb, lol) should mind it.

If we pause the program we get a summary of the search results.

```

iterations: 748371, time: 5.039 sec

value: 0.20

P1 emprical - BodySlam:0.97 Reflect:0.01 Rest:0.01 Earthquake:0.01

P1 nash - BodySlam:1.00 Reflect:0.00 Rest:0.00 Earthquake:0.00

P2 emprical - Psychic:0.98 Rest:0.01 Amnesia:0.01

P2 nash - Psychic:0.00 Rest:0.00 Amnesia:1.00

matrix:

Psychic Rest Amnesia

BodySlam 0.20 0.25 0.20

Reflect 0.10 0.14 0.12

Rest 0.09 0.19 0.15

Earthqua 0.14 0.19 0.19

```

The 'value' is the expected value for player 1 at the root position we've entered. Player 1 wins are given a value of 1.0, losses 0.0, and draws 0.5. This means that the Slowbro significantly favored here (4:1 odds of winning).

The bottom portion is the 'empirical matrix'. During each MCTS iteration, both players use their bandit algorithms to select an action. The pairs of actions define a matrix and each iteration will update an entry of the matrix, which stores the average (P1) value. This average includes the chance element of the game (e.g., the average value for the pair `(BodySlam, BodySlam)` in the position `Snorlax BodySlam 1% | Snorlax BodySlam 1%` is .5, since the outcome is decided by speed ties and 255 misses.)

Lastly we have the search policies. The empirical policies are just the number of times the bandit algorithm chose that action at the root node, divided by the number of iterations. As a result it almost always have non-zero probability for each action, even terrible ones. The 'Nash' policy is more refined. It is produces by solving Nash equilibrium to the empirical matrix.

```

The following might be helpful for understanding the 'Nash' aspect if the reader is familiar with vanilla MCTS setups.

In the famous RL algorithm AlphaZero, the search uses a UCB-backed MCTS. During self-play training, the algorithm selects moves in proportion to the empirical policy. This is to balance playing good moves with exploration. However during the testing phase (playing vs Stockfish) the moves with the highest average value were selected.

The policy given by solving the emprical value matrix is the '2D' analog of the greedy policy used in the testing phase of AlphaZero.

```

With the above in mind we can interpret the output of this very brief search.

Most of iterations were spent by P1 on Body Slam, which makes sense since it's wins are likely achieved with full paralysis. No other moves make sense as well. Similarly P2 spent most of the 750k iterations on Psychic. It has a chance of spc drop which will let Slowbro break though rest spam.

But notice that the P2 Nash strategy is to always select Amnesia. This makes sense since Slowbro can immediately start breaking without fishing for drops via Psychic. This was determined by solving the empirical matrix even though Amnesia was not selected nearly as many times.

This 1v1 is simple enough that we can interpret the players Nash strategies as the game theoretic solutions. We can support this idea by allowing the search to continue. The program informs us

```

(If the next input is a P1 choice index the battle is advanced; otherwise search is resumed.):

```

Simply hitting enter will resume the search. We can pause it again to get another output print.

```
iterations: 9710356, time: 75.81 sec

value: 0.24

P1 emprical - BodySlam:0.98 Reflect:0.01 Rest:0.01 Earthquake:0.01

P1 nash - BodySlam:1.00 Reflect:0.00 Rest:0.00 Earthquake:0.00

P2 emprical - Psychic:0.84 Rest:0.01 Amnesia:0.15

P2 nash - Psychic:0.00 Rest:0.00 Amnesia:1.00

matrix:

Psychic Rest Amnesia

BodySlam 0.24 0.27 0.24

Reflect 0.11 0.14 0.10

Rest 0.09 0.16 0.11

Earthqua 0.15 0.20 0.16

```

This information includes the information from the first search period, e.g. the first 748,371 iterations are counted among the second 9,710,356 iterations, etc.

The battle position is advanced in two different ways. If one valid choice index for player 1 in entered, the choice index for player 2 will be sampled from that player's Nash policy. The user can also enter choice indices for both players separated by a space. If the user plays out the battle, they may find that first turn Value estimate was pessimistic for Snorlax.

## generate

There is not