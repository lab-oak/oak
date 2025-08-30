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
constexpr auto in_dim = species_move_list_size;
constexpr auto out_dim = in_dim;

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

std::vector<Action> legal_actions(const auto &team) {
  std::vector<Action> actions;
  actions.reserve(max_actions);
}

} // namespace Team
}; // namespace Encode