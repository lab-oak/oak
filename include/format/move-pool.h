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
using PKMN::Data::Move;

template <typename F> consteval auto get_move_pool_sizes() {
  std::array<uint8_t, n_species> move_pool_sizes{};
  for (auto species = 0; species < n_species; ++species) {
    auto size = 0;
    for (auto m : F::LEARNSETS[species]) {
      size += m;
    }
    move_pool_sizes[species] = size;
  }
  return move_pool_sizes;
}

template <typename F> consteval auto get_max_move_pool_size() {
  constexpr auto size = get_move_pool_sizes<F>();
  constexpr auto max = *std::max_element(size.begin(), size.end());
  return max;
}

template <typename F> consteval auto get_move_pools_flat() {
  constexpr auto max = get_max_move_pool_size<F>();
  using Pool = std::array<Move, max>;
  std::array<Pool, n_species> list{};
  for (auto i = 0; i < n_species; ++i) {
    auto index = 0;
    for (auto m = 0; m < n_moves; ++m) {
      if (F::LEARNSETS[i][m]) {
        list[i][index++] = static_cast<Move>(m);
      }
    }
  }
  return list;
}

template <typename F> struct MovePool {
  static constexpr auto sizes = get_move_pool_sizes<F>();
  static constexpr auto pools = get_move_pools_flat<F>();
  static constexpr auto max_size = get_max_move_pool_size<F>();
  static constexpr auto size(const auto species) {
    return sizes[static_cast<uint8_t>(species)];
  }
  static constexpr const auto &get(const auto species) {
    return pools[static_cast<uint8_t>(species)];
  }
};

} // namespace Format