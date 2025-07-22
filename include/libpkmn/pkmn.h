#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/init.h>

namespace PKMNDetail {

template <typename T> auto get(T *t) { return t; }

template <typename T>
  requires(!std::is_pointer_v<T>)
auto get(const T &t) {
  return &t;
}
} // namespace PKMNDetail

namespace PKMN {

struct Set {
  static constexpr std::array<uint8_t, 4> max_pp{255, 255, 255, 255};

  Data::Species species;
  std::array<Data::Move, 4> moves;

  std::array<uint8_t, 4> pp{max_pp};
  float hp = 100 / 100;
  uint8_t status = 0;
  uint8_t sleeps = 0;
  Init::Boosts boosts = {};
  uint8_t level = 100;
  constexpr bool operator==(const Set &) const noexcept = default;
};

using Team = std::array<Set, 6>;

constexpr auto battle(const auto &p1, const auto &p2,
                      uint64_t seed = 0x123445) {
  pkmn_gen1_battle battle{};
  Init::init_side(p1, battle.bytes);
  Init::init_side(p2, battle.bytes + Layout::Sizes::Side);
  auto *ptr_64 =
      std::bit_cast<uint64_t *>(battle.bytes + Layout::Offsets::Battle::turn);
  ptr_64[0] = 0; // turn, last used, etc
  ptr_64[1] = seed;
  return battle;
}

constexpr pkmn_gen1_chance_durations durations() { return {}; }

constexpr auto durations(const auto &p1, const auto &p2) {
  pkmn_gen1_chance_durations durations{};
  auto &dur = View::ref(durations);
  Init::init_duration(p1, dur.get(0));
  Init::init_duration(p2, dur.get(1));
  return durations;
}

constexpr pkmn_gen1_battle_options options() { return {}; }

void set(pkmn_gen1_battle_options &options, const auto &log, const auto &chance,
         const auto &calc) {
  using PKMNDetail::get;
  return pkmn_gen1_battle_options_set(&options, get(log), get(chance),
                                      get(calc));
}

void set(pkmn_gen1_battle_options &options) {
  return pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
}

[[nodiscard]] pkmn_result update(pkmn_gen1_battle &battle, const auto c1,
                                 const auto c2,
                                 pkmn_gen1_battle_options &options) {
  const auto get_choice = [](const auto c, const uint8_t *side) -> pkmn_choice {
    using Choice = std::remove_cv<decltype(c)>::type;
    if constexpr (std::is_same_v<Choice, Data::Species>) {
      for (uint8_t i = 1; i < 6; ++i) {
        const auto id = side[Layout::Offsets::Side::order + i] - 1;
        if (static_cast<uint8_t>(c) ==
            side[24 * id + Layout::Offsets::Pokemon::species]) {
          return ((i + 1) << 2) | 2;
        }
      }
      throw std::runtime_error{"PKMN::update - invalid switch"};
    } else if constexpr (std::is_same_v<Choice, Data::Move>) {
      for (uint8_t i = 0; i < 4; ++i) {
        if (static_cast<uint8_t>(c) ==
            side[Layout::Offsets::Side::active +
                 Layout::Offsets::ActivePokemon::moves + 2 * i]) {
          return ((i + 1) << 2) | 1;
        }
      }
      throw std::runtime_error{"PKMN::update - invalid move"};
    } else if constexpr (std::is_integral_v<Choice>) {
      return c;
    } else {
      assert(false);
    }
  };
  pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  return pkmn_gen1_battle_update(
      &battle, get_choice(c1, battle.bytes),
      get_choice(c2, battle.bytes + Layout::Sizes::Side), &options);
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

} // namespace PKMN
