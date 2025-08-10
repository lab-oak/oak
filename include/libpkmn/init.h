#pragma once

#include <libpkmn/data.h>
#include <libpkmn/data/boosts.h>
#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/data/types.h>
#include <libpkmn/layout.h>
#include <util/random.h>

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace Init {

constexpr uint16_t compute_stat(uint8_t base, bool hp = false,
                                uint8_t level = 100) {
  const uint16_t evs = 255;
  const uint32_t core = (2 * (base + 15)) + 63;
  const uint32_t factor = hp ? level + 10 : 5;
  return core * level / 100 + factor;
}

constexpr PKMN::Pokemon init_pokemon(const auto &set) {
  PKMN::Pokemon pokemon{};
  // species
  pokemon.species = set.species;
  if (pokemon.species == Data::Species::None) {
    return pokemon;
  }
  // level
  if constexpr (requires { set.level; }) {
    assert(pokemon.level >= 1 && pokemon.level <= 100);
    pokemon.level = set.level;
  } else {
    pokemon.level = 100;
  }
  // stats
  const auto base_stats = Data::get_species_data(pokemon.species).base_stats;
  pokemon.stats.hp = compute_stat(base_stats[0], true, pokemon.level);
  pokemon.stats.atk = compute_stat(base_stats[1], false, pokemon.level);
  pokemon.stats.def = compute_stat(base_stats[2], false, pokemon.level);
  pokemon.stats.spe = compute_stat(base_stats[3], false, pokemon.level);
  pokemon.stats.spc = compute_stat(base_stats[4], false, pokemon.level);
  // moves
  for (auto m = 0; m < 4; ++m) {
    if constexpr (std::is_convertible_v<decltype(set.moves[0]), Data::Move>) {
      pokemon.moves[m].id = static_cast<Data::Move>(set.moves[m]);
      pokemon.moves[m].pp = get_max_pp(pokemon.moves[m].id);
    } else {
      pokemon.moves[m] = set.moves[m];
      pokemon.moves[m].pp =
          std::min(set.moves[m].pp, get_max_pp(pokemon.moves[m].id));
    }
  }
  // hp
  if constexpr (requires { set.hp; }) {
    uint16_t hp;
    if constexpr (std::is_floating_point_v<decltype(set.hp)>) {
      hp = set.hp * pokemon.stats.hp;
    } else {
      hp = set.hp;
    }
    pokemon.hp = std::min(hp, pokemon.stats.hp);
  } else {
    pokemon.hp = pokemon.stats.hp;
  }
  // status
  if constexpr (requires { set.status; }) {
    pokemon.status = static_cast<Data::Status>(set.status);
  }
  // types
  const auto types = get_types(pokemon.species);
  pokemon.types =
      static_cast<uint8_t>(types[0]) | (static_cast<uint8_t>(types[1]) << 4);
  return pokemon;
}

constexpr uint16_t boost(uint16_t stat, int b) {
  const auto &pair = Data::boosts[b + 6];
  return std::min(999, stat * pair[0] / pair[1]);
}

constexpr PKMN::ActivePokemon init_active(const auto &set,
                                          const PKMN::Pokemon &pokemon) {
  // turn 0
  PKMN::ActivePokemon active{};
  active.stats = pokemon.stats;
  active.species = pokemon.species;
  active.types = pokemon.types;
  active.moves = pokemon.moves;

  // boosts
  if constexpr (requires { set.boosts.atk; }) {
    active.stats.atk = boost(pokemon.stats.atk, set.boosts.atk);
    active.boosts.set_atk(set.boosts.atk);
  }
  if constexpr (requires { set.boosts.def; }) {
    active.stats.def = boost(pokemon.stats.def, set.boosts.def);
    active.boosts.set_def(set.boosts.def);
  }
  if constexpr (requires { set.boosts.spe; }) {
    active.stats.spe = boost(pokemon.stats.spe, set.boosts.spe);
    active.boosts.set_spe(set.boosts.spe);
  }
  if constexpr (requires { set.boosts.spc; }) {
    active.stats.spc = boost(pokemon.stats.spc, set.boosts.spc);
    active.boosts.set_spc(set.boosts.spc);
  }

  // de-facto stats
  if constexpr (requires { set.stats.atk; }) {
    if (set.stats.atk) {
      active.stats.atk = set.stats.atk;
    }
  }
  if constexpr (requires { set.stats.def; }) {
    if (set.stats.def) {
      active.stats.def = set.stats.def;
    }
  }
  if constexpr (requires { set.stats.spe; }) {
    if (set.stats.spe) {
      active.stats.spe = set.stats.spe;
    }
  }
  if constexpr (requires { set.stats.spc; }) {
    if (set.stats.spc) {
      active.stats.spc = set.stats.spc;
    }
  }

  // volatiles
  // TODO

  return active;
}

constexpr PKMN::Side init_side(const auto &sets) {
  PKMN::Side side{};

  bool first_alive = true;
  for (auto i = 0; i < sets.size(); ++i) {
    const auto pokemon = init_pokemon(sets[i]);
    if (pokemon.hp) {
      side.order[i] = i + 1;
      if (first_alive) {
        side.active = init_active(sets[i], pokemon);
        first_alive = false;
      }
    }
    side.pokemon[i] = pokemon;
  }

  return side;
}

constexpr void init_duration(const auto &party, PKMN::Duration &duration) {
  if constexpr (requires { party.pokemon; }) {
    init_party(party.pokemon, duration);
  }
  const auto n = party.size();
  assert(n <= 6);
  for (auto i = 0; i < n; ++i) {
    const auto &pokemon = party[i];
    if constexpr (requires { pokemon.sleeps; }) {
      if (Data::is_sleep(pokemon.status) && !Data::self(pokemon.status)) {
        duration.set_sleep(i, pokemon.sleeps);
      }
    }
  }
}

static_assert(compute_stat(100, false) == 298);
static_assert(compute_stat(250, true) == 703);
static_assert(compute_stat(5, false) == 108);

struct Boosts {
  int atk;
  int def;
  int spe;
  int spc;

  bool operator==(const Boosts &) const noexcept = default;
};
} // namespace Init
