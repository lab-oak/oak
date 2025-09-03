#pragma once

#include <libpkmn/data/moves.h>

#include <array>

/*

Sytactic sugar over the raw tables. Also improves performance w.r.t. scanning
(e.g. writing TrjaectoryInput) since the movesets are sparse.

*/

namespace Format {

constexpr int n_species = PKMN::Data::all_species.size();
constexpr int n_moves = PKMN::Data::all_moves.size();
using LearnsetTable = std::array<std::array<bool, n_moves>, n_species>;
using PKMN::Data::Move;

// struct Data {
//   LearnsetTable learnset;
// };

template <LearnsetTable ls> consteval auto get_move_pool_sizes() {
  std::array<uint8_t, n_species> move_pool_sizes{};
  for (auto species = 0; species < n_species; ++species) {
    auto size = 0;
    for (auto m : ls[species]) {
      size += m;
    }
    move_pool_sizes[species] = size;
  }
  return move_pool_sizes;
}

template <LearnsetTable ls> consteval auto get_max_move_pool_size() {
  constexpr auto size = get_move_pool_sizes<ls>();
  constexpr auto max = *std::max_element(size.begin(), size.end());
  return max;
}

template <LearnsetTable ls> consteval auto get_move_pools_flat() {
  constexpr auto max = get_max_move_pool_size<ls>();
  using Pool = std::array<Move, max>;
  std::array<Pool, n_species> list{};
  for (auto i = 0; i < n_species; ++i) {
    auto index = 0;
    for (auto m = 0; m < n_moves; ++m) {
      if (ls[i][m]) {
        list[i][index++] = static_cast<Move>(m);
      }
    }
  }
  return list;
}

template <LearnsetTable LEARNSETS> struct MovePool {
  static constexpr auto sizes = get_move_pool_sizes<LEARNSETS>();
  static constexpr auto pools = get_move_pools_flat<LEARNSETS>();
  static constexpr auto max_size = get_max_move_pool_size<LEARNSETS>();
  static constexpr auto size(const auto species) {
    return sizes[static_cast<uint8_t>(species)];
  }
  template <typename T> static constexpr const auto &get(const T species) {
    return pools[static_cast<uint8_t>(species)];
  }
};

} // namespace Format