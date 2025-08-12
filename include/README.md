# libpkmn/

The `libpkmn` C API is too barebones for anything beyond the example. It is basically incumbent on the user to replicate some of the zig interface themselves.

## libpkmn/data/

* data/boosts.h
* data/moves.h
* data/species.h
* data/types.h
* data/status.h

Recreating basic `libpkmn` types. In the names of the sleep status conditions, the number at the end represents the stored de-factor duration, e.g. `Sleep1` will wake up upon trying to use a move.

* strings.h

Here we hardcode species/move/type names as char arrays. This is so that we can make the data constexpr, as many compilers still do not support constexpr std::string.

* data.h

The composite data types of `libpkmn` such as `Battle`, `Side`, `Pokemon`, `ActivePokemon` are defined here.  This way members can be accessed directly; `pokemon.level` is much more clear and convenient than `battle.bytes[24 * i + 21]`.

In the remaining code, these structs are only used as references, not constructed as is. We use the overloaded `View::ref(auto&)` function which gives us a 'view' into the underlying byte buffer.

* layout.h

Duplication of the information in `libpkmn`'s layout.json. No magic constants!

* init.h

Helpers for initializing `Battle` and `Durations` structs.

*  rng.h

Pokemon-Showdown rng.

* strings.h

String/print functions for composite types, similar to `libpkmn/data/strings.h`. In particualar `battle_data_to_string` is the only way for a human to see what's going on in a battle.

# data/ 

Format specific data

* teams.h

This is simply all the Smogon RBY OU sample teams, as well as some 'legacy' for benchmark consistency. It is recommended that users replace or augement the data in this file with larger colections of teams. There is a utility `dev/convert.py` for converting Pokemon Showdowns "packed" format into this header.

* legal-moves.h

This hardcoded data tells us whether a pokemon can learn any given move by level 100. RBY does in fact have a few illegal combinations of moves, but not nearly as later generations with egg/event moves.

* move-pools.h

#  search/

Perfect info search code

#  train/

Value and policy targets, not specific to any network

# encode/

Oak specific battle and team-gen network encodings

# nn/

NN implementation. Full precision and quantized