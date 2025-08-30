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

// probably should not be used, instead incrementally update input
void write(const auto &team, float *const t) {
  for (const auto &set : team) {
    if (static_cast<bool>(set.species)) {
      t[species_move_table(set.species, 0)] = 1;
    }
    for (const auto move : set.moves) {
      if (static_cast<bool>(move)) {
        t[species_move_table(set.species, move)] = 1;
      }
    }
  }
}

void write_policy_mask_flat(const auto &team, auto *t) {
  auto *t0 = t;
  assert(t0);
  bool needs_species = false;
  for (const auto &set : team) {

    if (!static_cast<bool>(set.species)) {
      needs_species = true;
      continue;
    }

    auto n_moves = 0;
    for (const auto move : set.moves) {
      n_moves += static_cast<bool>(move);
    }
    if (n_moves >= std::min(Format::move_pool_size(set.species), (uint8_t)4)) {
      continue;
    }

    auto pool = move_pool(set.species);
    // remove all moves already in the set
    for (const auto move : set.moves) {
      if (static_cast<bool>(move)) {
        for (auto &m : pool) {
          if (m == move) {
            m = PKMN::Data::Move::None;
            break;
          }
        }
      }
    }
    // write remaining moves
    for (const auto move : pool) {
      if (static_cast<bool>(move)) {
        *t++ = species_move_table(set.species, move);
      }
    }
  }
  if (needs_species) {
    std::array<bool, 152> not_available{};
    for (const auto &set : team) {
      if (static_cast<bool>(set.species)) {
        not_available[static_cast<uint8_t>(set.species)] = true;
      }
    }
    for (auto i = 1; i <= 149; ++i) {
      if (not_available[i]) {
        continue;
      }
      *t++ = species_move_table(i, 0);
    }
  }
}

} // namespace Team
}; // namespace Encode