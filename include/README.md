# Warning: Incomplete/WIP

# libpkmn

The simulator is the core of any search library. We use `libpkmn` since it matches Showdown's behaviour and was designed with search in mind.

There are various comptime options when building libpkmn that need to be mentioned.

* `chance`/`calc`

Theese are the most important options by far since MCTS is not really possible without them. They solve the problem of 'randomizing' the state at the start of the iteration and allow for damage clamping. Because the cart and Showdown fundamentally rely on *hidden* variables (not *private* but rather unknown to either player) we cannot simply change the battle seed to sample transitions in the Markov sense, we must also sample the hidden variables based on public observations of sleep turns, wrap turns, etc. The damage roll clamping is necessary because there are 39 possible rolls in gen 1. 

* `log`

This enables log output like Showdown's omniscient log. We do not use this information for search even though that requires us to somehow keep track of what happened during a battle update to keep the game in sync with the tree. Although the log would work for this, the `pkmn_gen1_chance_actions` from the `chance` option is better for this.

The following options are minor in comparison:

* `ebc=false`

Competitive teams are not in danger of endless battles, so disabling improves performance

* `miss=false`

The infamous "255" is patched for no reason other that than its too unlikely to meaningfully impact search behaviour or output.

* `advance=false`

This option is only needed to exactly rng match, but as long as the sampling behaviour is the same we don't care.

* `key=true`

Just an improvement for the `chance` flag.

From now until stated otherwise, the sections refer to directories in the `include` folder and subsections will refer to files.. 

# libpkmn/data/

These headers just deplicate the Move, Speices, Types, Status enums and some tables like boosts and the type chart. Necessary quality of life.

## data.h

Here we redefine the real data: `Pokemon`, `ActivePokemon`, `Side`, `Battle`. These structs are not actually used for battle updates, which is the real logic and difficulty of a simulator. We use libpkmn's opaque `pkmn_gen1_battle_update` function for that. Instead the structs are used to fill in the functionality not provided by libpkmn, like initializing a battle or randomizing the hidden variables.

Regarding C++, we never actually initalize a `PKMN::Pokemon` etc object, instead we use `reinterpret_cast` to view a `pkmn_gen1_battle` and `pkmn_gen1_chance_durations` as `PKMN::Battle` and `PKMN::Durations`, resp. This spares us a lot of unreadable 'bit poking` and probably does not have a performance penalty.

## layout.h

More names for magic constants.

## init.h

Constrcting a battle requires little work on our end, we only need to compute base stats. However the use case of an analysis engine requires us to construct battles that are in progress and with as little tiresome specification as possible (e.g. "level is 100").

This motivates the `Init::Pokemon` struct

```cpp
struct Pokemon {
  Species species;
  std::array<Move, 4> moves;
  std::array<uint8_t, 4> pp = {64, 64, 64, 64};
  // actual hp value, ignored if negative
  int hp = -1;
  // percent
  uint percent = 100;
  Status status = Status::None;
  uint8_t sleeps = 0;
  uint8_t level = 100;
};
```

This data is more of a default description of a pokemon. It only requires species and moves, all other values are default and effectively describe a fresh slot.

> `percent` is the expected input since its rare for a use to have an exact hp value. The `hp` is only used when its value is non negative and it will override `percent` in that case

> `sleeps` is where the sleep *duration* (the number of turns slept as reported by the Showdown cliet) are entered


## strings.h

Just debug and quality of life functions.

The most important function is `battle_data_to_string` which prints a battle like:

```
Alakazam: 100% (313/313) Psychic:16 SeismicToss:32 ThunderWave:31 Recover:32 
  Starmie: 100% (323/323) Surf:24 Thunderbolt:24 ThunderWave:32 Recover:32 
  Rhydon: 100% (413/413) Earthquake:16 BodySlam:24 Substitute:16 RockSlide:16 
  Chansey: 100% (703/703) Sing:24 IceBeam:16 ThunderWave:32 SoftBoiled:16 
  Snorlax: 100% (523/523) BodySlam:24 Reflect:32 IceBeam:16 Rest:16 
  Tauros: 100% (353/353) BodySlam:24 HyperBeam:8 Blizzard:8 Earthquake:16 
--- --- --- 2 --- --- ---
(spe 158>>39) 
Snorlax: 100% (523/523) PAR BodySlam:24 Reflect:32 HyperBeam:8 Rest:16 
  Cloyster: 100% (303/303) Blizzard:8 Clamp:16 Explosion:8 Rest:16 
  Alakazam: 100% (313/313) Psychic:16 SeismicToss:32 ThunderWave:32 Recover:32 
  Chansey: 100% (703/703) IceBeam:16 Thunderbolt:24 ThunderWave:32 SoftBoiled:16 
  Jynx: 100% (333/333) LovelyKiss:16 Blizzard:8 Psychic:16 Rest:16 
  Tauros: 100% (353/353) BodySlam:24 HyperBeam:8 Blizzard:8 Earthquake:16 
```

## pkmn.h

This top level header for the C++ wrapper provides cleaner wrappers for the non-battle-update libpkmn functions.

# Teams

No functions here, only the current Smogon sample teams with some legacy benchmark teams

# Search

This project supports an 'Information Set Monte Carlo Tree Search' approach where imperfect information is handled in a two step process

* one

* two

In normal (alternating move) MCTS, each node stores data for a bandit algorithm, typically UCB/PUCB. This data is used to choose a policy for the forward phase of the iteration and then updated with the leaf-node value in the backward phase. Our approach for the simultaneous move case is termed "joint bandits": at each node we store separate bandit data for each player. Each player samples and updates their data simulateously and independently.

> Theory


## durations.h

# search/bandit/

* ucb

* exp3



## joint.h


## mcts.h

The MCTS code in Oak is difficult to parse at first because of its heavy use of compile time logic.

However, all variations are essentially the standard recursive formulation of mcts:

```cpp
float run_iteration(auto& node, auto &state) {
  if (node.is_expanded()) {
    // select and commit moves for both players
    auto obs = state.transition(p1_choice, p2_choice);
    auto& child = node.get_child(obs);
    // get corresponding child node
    auto value = run_iteration(child, state);
    // update node stats using leaf value
    return value;
  } else {
    if (state.is_terminal()) {
      return state.value();
    }
    // if state is not terminal, exapand node
    // and get value estimate from rollout/network inference/etc
    auto value = get_estimate();
    return value;
  }
}
```

During this forward the phase the state will be mutated with no way to return to the state it was originally at the root node. For this reason we must copy the state at the start of each iteration and perform the iteration on the copy instead.

### State

```cpp
struct Input {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
};
```

The so-called "state" refers of course to the libpkmn battle struct but also to the durations. This is explained here TODO.

The `pkmn_result` is just a single byte that tells us what is requested from either player at that point in the battle: whether they may move/switch, switch, or pass. Strictly speaking it is not necessary because that information can be deduced from the battle, but it is an argument to `pkmn_gen1_choices` and it is more effecient to store and update it (`result = pkmn_gen1_battle_update(...)`) than to construct it every update.

### Search Results

```cpp
struct Output {
  uint8_t m;
  uint8_t n;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;

  std::array<std::array<size_t, 9>, 9> visit_matrix;
  std::array<std::array<double, 9>, 9> value_matrix;
  double total_value;
  double empirical_value;
  double nash_value;
  std::array<double, 9> p1_prior;
  std::array<double, 9> p2_prior;
  std::array<double, 9> p1_empirical;
  std::array<double, 9> p2_empirical;
  std::array<double, 9> p1_nash;
  std::array<double, 9> p2_nash;

  size_t iterations;
  std::chrono::milliseconds duration;
};
```

All variations of the MCTS produce the exact same kind of output. It is meant to summarize the value of the state and the best move for either player. It also stores some information that is used during the search, so it is both a parameter and return type.

* `m`, `n`, `p1_choices`, `p2_choices`



* `visit_matrix`, `value_matrix`


* `iterations`, `duration`


### run()

The main mcts function template has the signature

```cpp
  Output run(auto &device, const auto budget, const auto &params, auto &heap,
             auto &model, const MCTS::Input &input, Output output = {})
```

#### Budget

There are 3 modes for the search budget

* iteration count

The search will run until this many iterations have been performed

* `std::chrono` duration

The search will run for at least this length of time

* pointer to `bool` like

This acts like an run/halt flag. The search will run until the value of the flag is observed once as `false`. 

#### Model

There are 3 kinds of value (and policy) estimators

* MonteCarlo

The struct holds no data and simply specifies that monte carlo rollouts should be used at leaf nodes

* NN::Battle::Network

The Oak battle network. It always provides a value but may also return policy inference in the form of logits. Whether it does so is determined by the bandit algorithm used

#### Params

The parameters ("c" for UCB, "gamma" and "alpha" for Exp3) of the bandit algorithms.

Each of these structs (e.g. `UCB::Params`, `PExp3::Params`) may be wrapped in a `MatrixUCB<typename T>` template that specifies that the MatrixUCB algorithm should be used at the root node. More on that below

#### Heap

The type of data structure used to store the search stats

* Node

* Table


## hash.h

## poke-engine-evaluate.h

# encode/

# nn/

# nn/battle/

# nn/build/

