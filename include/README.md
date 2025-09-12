# Warning: Incomplete/WIP

# libpkmn/

The `libpkmn` C API is too barebones for a project of this scale. We write a C++ wrapper for the C bindings that basically mirrors the native Zig library as much as possible.

## libpkmn/data/

* data/boosts.h
* data/moves.h
* data/species.h
* data/types.h

These files basically just define the types or data they are named after. There may be some ancillary information like max PP, etc.

* data/status.h

The Status type is a bit more nuanced because of sleep.

* strings.h

Here we hardcode species/move/type names as char arrays. Recent ubuntu versions have had trouble with some later C++ features like `constexpr std::string`. We prefer to make things constexpr whenever possible, so we settle for `char[]` instead of `std::string`.

```cpp
move_string(151); // returns a string "Mew'.
```

* data.h

The composite data types of `libpkmn` such as `Battle`, `Side`, `Pokemon`, `ActivePokemon` are defined here.  This way members can be accessed directly; `pokemon.level` is much more clear and convenient than `battle.bytes[24 * i + 21]`.

In the remaining code, these structs are only used as references, not constructed as is. We use the overloaded `PKMN::view(auto&)` function which gives us a 'view' into the underlying byte buffer.

* layout.h

Duplication of the information in `libpkmn`'s layout.json. No magic constants!

* init.h

Helpers for initializing `Battle` and `Durations` structs. A battle is typically initialized from the start of the game using just the team information. In this case intializing is just computing the stats from base stats, setting pp, seeting the seed, and zero'ing certain fields.

This file also handles the extended of functionality of initializing an in progress battle. This means setting bits in the volatiles and durations structs. As always, the battle state is not omniscient so e.g. `duration.confusion` is set rather than `volatiles.confusionLeft`. That is done as needed with the `apply_durations` function.

*  rng.h

Pokemon-Showdown rng. Very fast but low quality - especially if not seeded correctly.

* strings.h

String/print functions for composite types, similar to `libpkmn/data/strings.h`. In particualar `battle_data_to_string` is the only way for a human to see what's going on in a battle.

# data/ 

Format specific data

* teams.h

This is simply all the Smogon RBY OU sample teams, as well as some 'legacy' for benchmark consistency. It is recommended that users replace or augement the data in this file with larger colections of teams. There is a utility `dev/convert.py` for converting Pokemon Showdowns "packed" format into this header but that should not be necessary now that `util/parse` can dynamically load teams from disk.

* legal-moves.h

This hardcoded data tells us whether a pokemon can learn any given move by level 100. RBY does in fact have a few illegal combinations of moves, but not nearly as later generations with egg/event moves. We simply ignore the previous fact, meaning illegal sets like Stun Spore + Stomp Executor are possible in the team generation code. The player consensus is that the banned movesets are not competitively relevant.

* move-pools.h

 Using the previous data, this is a more user-friendly `move_pool(species)` function that returns an a `Move::None` padded array of that pokemons legal moves.

#  search/

* durations.h

The bandit algorithms are implemented in the typical 'single-player' fashion and the combined with the `Joint<>` helper.

## search/bandits

* bandits/ucb.h
* bandits/ucb-policy.h

A single modification to the `c` weight of the UCB algorithm so that it is equivalent to `PUCB` with uniform policy. This is soley to better test the efficacy of policy network inference rather than optimizing playing strength. 

* bandits/exp3.h
* bandits/exp3-policy.h

The `PExp3` algorithm is ad hoc and should be tested and improved.

* joint.h

This pattern does not allow for implementation of 'matrix aware' algorithms like [MatrixUCB](). In practice the cost of computing nash equilibrium at each node was catastrophic for performance in time trials

* mcts.h

Lots to expliain TODO

* tree.h

A node is the joint bandit stats plus a std::map to child nodes. Child nodes are uniquely identified by the tuple of P1 actions, P2 actions, and `pkmn_gen1_chance_actions`. The chance actions are observations of how the turn played out: damage rolls, secondary procs, crits, etc.

This is the reason for the `--option=key` flag in the `Libpkmn` build script. Without this flag, the chance actions would reveal hidden info like sleep rolls.

#  train/

Data formats for training neural networks. The code here is all the stuff that it is independent of any particular NN/Encoding setup. This is why `compressed-trajectory.h` is in `include/encode`.

## train/battle

Generally speaking, these formats support reinforcement setups like AlphaZero, LeelaChess0, etc.

* battle/target.h

Value and policy targets. We 

* battle/frame.h


## train/build

We assume that team building is a process where the actions are arbitrary 'diffs' between one team to another.

This means an action could be 'recove the snorlax and add chansey with softboiled and remove FireBlast from Tauros'

# encode/

Oak specific battle and team-gen network encodings

# nn/

NN implementation. Full precision and quantized