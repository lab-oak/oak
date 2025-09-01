#pragma once

#include <format/move-pools.h>
#include <libpkmn/data.h>
#include <libpkmn/data/status.h>
#include <libpkmn/pkmn.h>
#include <train/build-trajectory.h>

#include <array>
#include <cassert>
#include <functional>

namespace Encode {

namespace Team {

using Train::TeamBuilding::Action;
using Train::TeamBuilding::BasicAction;

consteval auto get_species_move_list_size() {
  uint16_t size = 0;
  for (auto s = 1; s <= 149; ++s) {
    for (auto m = 0; m < 166; ++m) {
      if (Format::MOVE_POOLS[s][m] || !m) {
        ++size;
      }
    }
  }
  return size;
}

constexpr auto species_move_list_size = get_species_move_list_size();
constexpr auto n_dim = species_move_list_size;

// max number of actions when rolling out the build network
consteval int get_max_actions() {
  int n = 0;
  auto stored = Format::MOVE_POOL_SIZES;
  std::sort(stored.begin(), stored.end(), std::greater<uint8_t>());
  for (auto i = 0; i < 5; ++i) {
    n += (int)stored[i];
  }
  n += 151 - 5;
  return n;
}

constexpr int max_actions = get_max_actions();

consteval auto get_species_move_data() {
  std::array<std::array<uint16_t, 166>, 152> table{};
  std::array<std::pair<uint8_t, uint8_t>, species_move_list_size> list{};

  uint16_t index = 0;
  for (auto s = 1; s <= 149; ++s) {
    for (auto m = 0; m < 166; ++m) {
      if (Format::MOVE_POOLS[s][m] || !m) {
        table[s][m] = index;
        list[index] = {s, m};
        ++index;
      }
    }
  }
  return std::pair<decltype(table), decltype(list)>{table, list};
}

constexpr auto SPECIES_MOVE_DATA = get_species_move_data();
constexpr auto SPECIES_MOVE_TABLE = SPECIES_MOVE_DATA.first;
constexpr auto SPECIES_MOVE_LIST = SPECIES_MOVE_DATA.second;

inline constexpr auto species_move_table(const auto species, const auto move) {
  return SPECIES_MOVE_TABLE[static_cast<uint8_t>(species)]
                           [static_cast<uint8_t>(move)];
}

inline constexpr auto species_move_list(const auto index) {
  return SPECIES_MOVE_LIST[index];
}

constexpr auto get_legal_species() {
  std::array<Species, 151> legal_species{};
  for (auto s = 1; s < 150; ++s) {
    legal_species[s - 1] = static_cast<Species>(s);
  }
  return legal_species;
}

int action_index(const Action &action) {
  if (action.size() != 1) {
    throw std::runtime_error{"Compound actions not supported."};
    return -1;
  }
  const auto species = action.species;
  const auto move = action.move;
  if (action.swap > 0) {
    auto index = species_move_table(species, 0);
    return index;
  }
  auto index = species_move_table(species, move);
  assert(index > 0 || (species == Species::Bulbasaur && move == Move::None));
  return index;
}

std::array<float, n_dim> write(const auto &team) {
  std::array<float, n_dim> input{};
  return input;
}

std::vector<Action> legal_actions(const auto &team) {
  using PKMN::Data::Move;
  using PKMN::Data::Species;
  std::vector<Action> actions;
  actions.reserve(max_actions);

  std::array<Species, 151> legal_species{};
  auto species_end = legal_species + 149;
  bool need_species =
      std::any_of(team.begin(), team.end(),
                  [](const auto &set) { return set.species == Move::None; });
  if (need_species) {
    legal_species = get_legal_species();
  }

  for (auto i = 0; i < team.size(); ++i) {
    const auto &set = team[i];
    if (set.species != Species::None) {
      species_end =
          std::remove(legal_species.start(), species_end, set.species);

      auto empty = std::find(set.moves.start(), set.moves.end(), Move::None);

      if (empty != set.moves.end()) {
        auto move_pool = Format::move_pool(set.species);
        const auto start = move_pool.start();
        auto end = start + move_pool_size(set.species);
        for (auto j = 0; j < set.moves; ++j) {
          const auto move = set.moves[j];
          if (move.id == Move::None && start != end) {
            end = std::remove(start, end, move.id);
          }
        }
        actions.emplace_back({BasicAction{}})
      }
    }
  }

  return actions;
}

} // namespace Team

} // namespace Encode
