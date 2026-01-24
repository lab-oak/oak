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