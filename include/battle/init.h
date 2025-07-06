#pragma once

#include <battle/view.h>
#include <data/layout.h>
#include <data/moves.h>
#include <data/species.h>
#include <data/status.h>
#include <data/types.h>
#include <util/random.h>

#include <assert.h>
#include <bit>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace Layout;

using Data::get_species_data;
using Data::get_types;
using Data::Move;
using Data::Species;
using Data::Status;

constexpr uint16_t compute_stat(uint8_t base, bool hp = false) {
  const uint16_t evs = 255;
  const uint32_t core = (2 * (base + 15)) + 63;
  return hp ? core + 110 : core + 5;
}

constexpr std::array<uint16_t, 5> compute_stats(const auto &pokemon) {
  std::array<uint16_t, 5> stats;
  const auto base = get_species_data(pokemon.species).base_stats;
  const auto ev = [&pokemon]() {
    if constexpr (requires { pokemon.ev; }) {
      return pokemon.ev;
    } else {
      return std::array<uint8_t, 5>{63, 63, 63, 63, 63};
    }
  }();
  const auto dv = [&pokemon]() {
    if constexpr (requires { pokemon.dv; }) {
      return pokemon.dv;
    } else {
      return std::array<uint8_t, 5>{15, 15, 15, 15, 15};
    }
  }();
  for (int s = 0; s < 5; ++s) {
    const uint32_t core = 2 * (base[s] + dv[s]) + ev[s];
    stats[s] = (s == 0) ? core + 110 : core + 5;
  }
  return stats;
}

constexpr void init_pokemon(const auto &pokemon, uint8_t *const bytes,
                            uint8_t *const duration_bytes, auto i = 0) {
  const auto species = pokemon.species;
  if (species == Species::None) {
    return;
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

    if constexpr (requires { pokemon.sleeps; }) {
      if (Data::is_sleep(pokemon.status) && !Data::self(pokemon.status)) {
        auto &d = *reinterpret_cast<View::Duration *>(duration_bytes);
        d.set_sleep(i, pokemon.sleeps);
      }
    }
  } else {
    bytes[20] = 0;
  }
  bytes[21] = static_cast<uint8_t>(species);
  const auto types = get_types(species);
  bytes[22] =
      (static_cast<uint8_t>(types[1]) << 4) | static_cast<uint8_t>(types[0]);
  if constexpr (requires { pokemon.level; }) {
    bytes[23] = pokemon.level;
  } else {
    bytes[23] = 100;
  }
}

constexpr std::array<std::array<uint8_t, 2>, 13> boosts{
    std::array<uint8_t, 2>{25, 100}, // -6
    {28, 100},                       // -5
    {33, 100},                       // -4
    {40, 100},                       // -3
    {50, 100},                       // -2
    {66, 100},                       // -1
    {1, 1},                          //  0
    {15, 10},                        // +1
    {2, 1},                          // +2
    {25, 10},                        // +3
    {3, 1},                          // +4
    {35, 10},                        // +5
    {4, 1}                           // +6
};

constexpr uint16_t boost(uint16_t stat, int b) {
  const auto &pair = boosts[b + 6];
  return std::min(999, stat * pair[0] / pair[1]);
}

constexpr void init_active(const auto &active, uint8_t *const bytes) {
  auto &pokemon = *reinterpret_cast<View::Pokemon *>(bytes);
  auto &active_pokemon =
      *reinterpret_cast<View::ActivePokemon *>(bytes + Offsets::Side::active);

  if constexpr (requires { active.volatiles; }) {
  }
  auto *stats = reinterpret_cast<uint16_t *>(bytes + Offsets::Side::active);
  if constexpr (requires { active.boosts.atk; }) {
    stats[1] = boost(pokemon.stats().atk(), active.boosts.atk);
    active_pokemon.set_boost_atk(active.boosts.atk);
  }
  if constexpr (requires { active.boosts.def; }) {
    stats[2] = boost(pokemon.stats().def(), active.boosts.def);
    active_pokemon.set_boost_def(active.boosts.def);
  }
  if constexpr (requires { active.boosts.spe; }) {
    stats[3] = boost(pokemon.stats().spe(), active.boosts.spe);
    active_pokemon.set_boost_spe(active.boosts.spe);
  }
  if constexpr (requires { active.boosts.spc; }) {
    stats[4] = boost(pokemon.stats().spc(), active.boosts.spc);
    active_pokemon.set_boost_spc(active.boosts.spc);
  }
}

constexpr void init_party(const auto &party, uint8_t *const bytes,
                          uint8_t *const duration_bytes) {
  const uint8_t n = party.size();
  assert(n > 0 && n <= 6);
  std::memset(bytes, 0, 24 * 6);
  std::memset(bytes + Offsets::Side::order, 0, 6);

  uint8_t n_alive = 0;

  for (uint8_t i = 0; i < n; ++i) {
    const auto &set = party[i];
    assert(set.moves.size() <= 4);
    init_pokemon(set, bytes + i * Sizes::Pokemon, duration_bytes, i);
    if (i == 0) {
      init_active(set, bytes);
    }

    if (set.species != Data::Species::None) {
      if constexpr (requires { set.hp; }) {
        if (set.hp == 0) {
          continue;
        }
      }
      bytes[Offsets::Side::order + n_alive] = i + 1;
      ++n_alive;
    }
  }
}

constexpr void init_side(const auto &side, uint8_t *const bytes,
                         uint8_t *const duration_bytes) {
  if constexpr (requires { side.pokemon; }) {
    init_party(side.pokemon, bytes, duration_bytes);
    if constexpr (requires { side.active; }) {
      init_active(side.active, bytes + Offsets::Side::active);
    }
  } else {
    init_party(side, bytes, duration_bytes);
  }
}
} // end anonymous namespace

namespace Init {

void apply_durations(auto &device, pkmn_gen1_battle &b,
                     const pkmn_gen1_chance_durations &d) {

  static constexpr std::array<std::array<uint8_t, 40>, 4> multi{
      std::array<uint8_t, 40>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                              2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4},
      std::array<uint8_t, 40>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                              2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
      std::array<uint8_t, 40>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
      std::array<uint8_t, 40>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  auto &battle = View::ref(b);
  const auto &durations = View::ref(d);
  for (auto s = 0; s < 2; ++s) {
    auto &side = battle.side(s);
    const auto &duration = durations.duration(s);

    auto &vol = side.active().volatiles();

    if (const auto binding = duration.binding()) {
      const auto index = device.random_int(40);
      vol.set_attacks(multi[binding - 1][index]);
    } else if (const auto attacking = duration.attacking()) {
      if (vol.bide()) {
        if (attacking == 3) {
          vol.set_attacks(1);
        } else {
          vol.set_attacks(4 - (attacking + device.random_int(2)));
        }
      } else {
        const auto index = device.random_int(40);
        vol.set_attacks(multi[attacking - 1][index]);
      }
    }
    if (const auto confusion = duration.confusion()) {
      const uint8_t max = 5 - confusion;
      vol.set_confusion_left(static_cast<uint8_t>(device.random_int(max) + 1));
    }
    if (const auto disable = duration.disable()) {
      const uint8_t max = 8 - disable;
      vol.set_disable_left(static_cast<uint8_t>(device.random_int(max) + 1));
    }

    for (auto p = 0; p < 6; ++p) {
      if (const auto sleep = duration.sleep(0)) {
        const auto slot = side.order(p) - 1;
        auto &pokemon = side.pokemon(slot);
        auto &status = reinterpret_cast<uint8_t &>(pokemon.status());

        if (!Data::is_sleep(status) || Data::self(status)) {
          continue;
        }

        const uint8_t max = 8 - sleep;
        status &= 0b11111000; // keep rest bit, clear sleep remaining
        status |= static_cast<uint8_t>(device.random_int(max) + 1);
      }
    }
  }
}

struct Boosts {
  int atk;
  int def;
  int spe;
  int spc;

  bool operator==(const Boosts &) const noexcept = default;
};

struct Set {
  Species species;
  std::array<Move, 4> moves;
  std::array<uint8_t, 4> pp{0xFF, 0xFF, 0xFF, 0xFF};
  float hp = 1;
  uint8_t status = 0;
  uint8_t sleeps = 0;
  Boosts boosts{};
  constexpr bool operator==(const Set &) const = default;
};

using Team = std::array<Set, 6>;

struct Config {
  std::array<Set, 6> pokemon;
};

// get durations from config, applly durations using ad hoc device
constexpr auto battle_data(const auto &p1, const auto &p2,
                           uint64_t seed = 0x123445) {
  pkmn_gen1_battle battle{};
  pkmn_gen1_battle_options options{};
  auto *durations_ptr = pkmn_gen1_battle_options_chance_durations(&options);
  init_side(p1, battle.bytes, durations_ptr->bytes);
  init_side(p2, battle.bytes + Sizes::Side, durations_ptr->bytes + 4);
  auto *ptr_64 =
      std::bit_cast<uint64_t *>(battle.bytes + Offsets::Battle::turn);
  ptr_64[0] = 0; // turn, last used, etc
  ptr_64[1] = seed;
  prng device{seed};
  Init::apply_durations(device, battle, *durations_ptr);
  return std::pair<pkmn_gen1_battle, pkmn_gen1_chance_durations>{
      battle, *durations_ptr};
}

constexpr pkmn_gen1_battle_options options() { return {}; }

[[nodiscard]] pkmn_result update(pkmn_gen1_battle &battle, const auto c1,
                                 const auto c2,
                                 pkmn_gen1_battle_options &options) {
  const auto get_choice = [](const auto c, const uint8_t *side) -> pkmn_choice {
    using Choice = decltype(c);
    if constexpr (std::is_same_v<Choice, Species>) {
      for (uint8_t i = 1; i < 6; ++i) {
        const auto slot = side[Offsets::Side::order + i] - 1;
        if (static_cast<uint8_t>(c) ==
            side[24 * slot + Offsets::Pokemon::species]) {
          return ((i + 1) << 2) | 2;
        }
      }
      throw std::runtime_error{"Init::update - invalid switch"};
    } else if constexpr (std::is_same_v<Choice, Move>) {
      for (uint8_t i = 0; i < 4; ++i) {
        if (static_cast<uint8_t>(c) ==
            side[Offsets::ActivePokemon::moves + 2 * i]) {
          return ((i + 1) << 2) | 1;
        }
      }
      throw std::runtime_error{"Init::update - invalid move"};
    } else if constexpr (std::is_integral_v<Choice>) {
      return c;
    } else {

      // static_assert(false);
    }
  };
  pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  return pkmn_gen1_battle_update(&battle, get_choice(c1, battle.bytes),
                                 get_choice(c2, battle.bytes + Sizes::Side),
                                 &options);
}

auto choices(const pkmn_gen1_battle &battle, const pkmn_result result)
    -> std::pair<std::vector<pkmn_choice>, std::vector<pkmn_choice>> {
  std::vector<pkmn_choice> p1_choices;
  std::vector<pkmn_choice> p2_choices;
  p1_choices.resize(PKMN_GEN1_MAX_CHOICES);
  p2_choices.resize(PKMN_GEN1_MAX_CHOICES);
  const auto m =
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
  const auto n =
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
  p1_choices.resize(m);
  p2_choices.resize(n);
  return {p1_choices, p2_choices};
}

auto score(const pkmn_result result) {
  switch (pkmn_result_type(result)) {
  case PKMN_RESULT_NONE: {
    return .5;
  }
  case PKMN_RESULT_WIN: {
    return 1.0;
  }
  case PKMN_RESULT_LOSE: {
    return 0.0;
  }
  case PKMN_RESULT_TIE: {
    return 0.5;
  }
  default: {
    assert(false);
    return 0.5;
  }
  }
}

auto score2(const pkmn_result result) {
  switch (pkmn_result_type(result)) {
  case PKMN_RESULT_NONE: {
    return 1;
  }
  case PKMN_RESULT_WIN: {
    return 2;
  }
  case PKMN_RESULT_LOSE: {
    return 0;
  }
  case PKMN_RESULT_TIE: {
    return 1;
  }
  default: {
    assert(false);
    return 1;
  }
  }
}

} // namespace Init

static_assert(compute_stat(100, false) == 298);
static_assert(compute_stat(250, true) == 703);
static_assert(compute_stat(5, false) == 108);
