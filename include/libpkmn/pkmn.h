#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/init.h>
#include <libpkmn/strings.h>

namespace PKMNDetail {

template <typename T> auto get_pointer(const T *t) { return t; }

template <typename T>
  requires(!std::is_pointer_v<T>)
auto get_pointer(const T &t) {
  return &t;
}
} // namespace PKMNDetail

namespace PKMN {

enum class Result : std::underlying_type_t<std::byte> {
  None = 0,
  Win = 1,
  Lose = 2,
  Tie = 3,
  Error = 4,
};

enum class Choice : std::underlying_type_t<std::byte> {
  Pass = 0,
  Move = 1,
  Switch = 2,
};

pkmn_result result(Result result = Result::None, Choice p1 = Choice::Move,
                   Choice p2 = Choice::Move) {
  return static_cast<uint8_t>(result) | (static_cast<uint8_t>(p1) << 4) |
         (static_cast<uint8_t>(p2) << 6);
}

struct Set {

  constexpr bool operator==(const Set &) const = default;

  Data::Species species;
  std::array<Data::Move, 4> moves;

  static constexpr std::array<uint8_t, 4> max_pp{255, 255, 255, 255};

  std::array<uint8_t, 4> pp = max_pp;
  float hp = 100 / 100;
  Data::Status status = Data::Status::None;
  uint8_t sleeps = 0;
  Init::Boosts boosts = {};
  uint8_t level = 100;
  Stats stats = {};
};

using Team = std::array<Set, 6>;

constexpr auto battle(const auto &p1, const auto &p2,
                      uint64_t seed = 0x123445) {
  PKMN::Battle battle{};
  battle.sides[0] = Init::init_side(p1);
  battle.sides[1] = Init::init_side(p2);
  battle.turn = 1;
  battle.rng = seed;
  return std::bit_cast<pkmn_gen1_battle>(battle);
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

auto &durations(pkmn_gen1_battle_options &options) {
  return *pkmn_gen1_battle_options_chance_durations(&options);
}

const auto &durations(const pkmn_gen1_battle_options &options) {
  return *pkmn_gen1_battle_options_chance_durations(&options);
}

void set(pkmn_gen1_battle_options &options, const auto &log, const auto &chance,
         const auto &calc) {
  using PKMNDetail::get_pointer;
  return pkmn_gen1_battle_options_set(&options, get_pointer(log),
                                      get_pointer(chance), get_pointer(calc));
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

auto choice_labels(const pkmn_gen1_battle &battle, const pkmn_result result)
    -> std::pair<std::vector<std::string>, std::vector<std::string>> {
  const auto [p1_choices, p2_choices] = choices(battle, result);
  std::vector<std::string> p1_labels{};
  std::vector<std::string> p2_labels{};
  for (auto i = 0; i < p1_choices.size(); ++i) {
    p1_labels.push_back(
        Strings::side_choice_string(battle.bytes, p1_choices[i]));
  }
  for (auto i = 0; i < p2_choices.size(); ++i) {
    p2_labels.push_back(Strings::side_choice_string(
        battle.bytes + Layout::Sizes::Side, p2_choices[i]));
  }
  return {p1_labels, p2_labels};
}

constexpr float score(const pkmn_result result) noexcept {
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

constexpr uint8_t score2(const pkmn_result result) noexcept {
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

} // namespace PKMN
