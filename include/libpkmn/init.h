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

constexpr std::array<uint16_t, 5> compute_stats(const auto &pokemon) {
  std::array<uint16_t, 5> stats;
  const auto base = Data::get_species_data(pokemon.species).base_stats;
  for (int s = 0; s < 5; ++s) {
    stats[s] = compute_stat(base[s], s == 0, pokemon.level);
  }
  return stats;
}

constexpr void init_pokemon(const auto &pokemon, uint8_t *const bytes,
                            auto i = 0) {
  const auto species = pokemon.species;
  if (species == Data::Species::None) {
    return;
  }
  if constexpr (requires { pokemon.level; }) {
    bytes[23] = pokemon.level;
  } else {
    bytes[23] = 100;
  }
  const auto stats = compute_stats(pokemon);
  auto *u16_ptr = std::bit_cast<uint16_t *>(bytes);
  for (int s = 0; s < 5; ++s) {
    u16_ptr[s] = stats[s];
  }

  auto *move_bytes = bytes + 10;
  std::array<uint8_t, 5> dv, ev;
  for (auto m = 0; m < 4; ++m) {
    const auto move = pokemon.moves[m];
    move_bytes[0] = static_cast<uint8_t>(move);
    move_bytes[1] = get_max_pp(move);
    if constexpr (requires { pokemon.pp; }) {
      move_bytes[1] = std::min(move_bytes[1], pokemon.pp[m]);
    }
    move_bytes += 2;
  }

  if constexpr (requires { pokemon.hp; }) {
    using HP = std::decay_t<decltype(pokemon.hp)>;
    if constexpr (std::is_floating_point_v<HP>) {
      u16_ptr[9] = std::min(std::max(pokemon.hp, HP{0}), HP{1}) * u16_ptr[0];
    } else {
      u16_ptr[9] = pokemon.hp;
    }
  } else {
    u16_ptr[9] = u16_ptr[0];
  }
  if constexpr (requires { pokemon.status; }) {
    bytes[20] = static_cast<uint8_t>(pokemon.status);
  } else {
    bytes[20] = 0;
  }
  bytes[21] = static_cast<uint8_t>(species);
  const auto types = get_types(species);
  bytes[22] =
      (static_cast<uint8_t>(types[1]) << 4) | static_cast<uint8_t>(types[0]);
}

constexpr uint16_t boost(uint16_t stat, int b) {
  const auto &pair = Data::boosts[b + 6];
  return std::min(999, stat * pair[0] / pair[1]);
}

constexpr void init_active(const auto &active, uint8_t *const bytes) {
  auto &pokemon = *reinterpret_cast<PKMN::Pokemon *>(bytes);
  auto &active_pokemon = *reinterpret_cast<PKMN::ActivePokemon *>(
      bytes + Layout::Offsets::Side::active);

  if constexpr (requires { active.volatiles; }) {
  }
  auto *stats =
      reinterpret_cast<uint16_t *>(bytes + Layout::Offsets::Side::active);
  if constexpr (requires { active.boosts.atk; }) {
    stats[1] = boost(pokemon.stats.atk, active.boosts.atk);
    active_pokemon.boosts.set_atk(active.boosts.atk);
  }
  if constexpr (requires { active.boosts.def; }) {
    stats[2] = boost(pokemon.stats.def, active.boosts.def);
    active_pokemon.boosts.set_def(active.boosts.def);
  }
  if constexpr (requires { active.boosts.spe; }) {
    stats[3] = boost(pokemon.stats.spe, active.boosts.spe);
    active_pokemon.boosts.set_spe(active.boosts.spe);
  }
  if constexpr (requires { active.boosts.spc; }) {
    stats[4] = boost(pokemon.stats.spc, active.boosts.spc);
    active_pokemon.boosts.set_spc(active.boosts.spc);
  }
}

constexpr void init_party(const auto &party, uint8_t *const bytes) {
  const uint8_t n = party.size();
  assert(n > 0 && n <= 6);
  std::memset(bytes, 0, 24 * 6);
  std::memset(bytes + Layout::Offsets::Side::order, 0, 6);

  uint8_t n_alive = 0;

  for (uint8_t i = 0; i < n; ++i) {
    const auto &set = party[i];
    assert(set.moves.size() <= 4);
    init_pokemon(set, bytes + i * Layout::Sizes::Pokemon, i);
    if (i == 0) {
      init_active(set, bytes);
    }
    if (set.species != Data::Species::None) {
      if constexpr (requires { set.hp; }) {
        if (set.hp == 0) {
          continue;
        }
      }
      bytes[Layout::Offsets::Side::order + n_alive] = i + 1;
      ++n_alive;
    }
  }
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

constexpr void init_side(const auto &side, uint8_t *const bytes) {
  if constexpr (requires { side.pokemon; }) {
    init_party(side.pokemon, bytes);
    if constexpr (requires { side.active; }) {
      init_active(side.active, bytes + Layout::Offsets::Side::active);
    }
  } else {
    init_party(side, bytes);
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
