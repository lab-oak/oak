#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>

#include <algorithm>
#include <array>

/*

Sytactic sugar over the raw tables. Also improves performance w.r.t. scanning
(e.g. writing TrjaectoryInput) since the movesets are sparse.

*/

namespace Format {

template <typename LS> consteval auto get_move_pool_sizes() {
  using PKMN::Data::all_species;
  std::array<uint8_t, all_species.size()> sizes{};
  std::transform(all_species.begin(), all_species.end(), sizes.begin(),
                 [](const auto species) {
                   const auto moves = LS::data[static_cast<uint8_t>(species)];
                   return std::count_if(moves.begin(), moves.end(),
                                        [](const bool b) { return b; });
                 });
  return sizes;
}

template <typename LS> consteval auto get_max_move_pool_size() {
  const auto sizes = get_move_pool_sizes<LS>();
  return *std::max_element(sizes.begin(), sizes.end());
}

template <typename LS> consteval auto get_move_pools() {
  using PKMN::Data::all_moves;
  using PKMN::Data::all_species;
  using PKMN::Data::Move;

  std::array<std::array<Move, get_max_move_pool_size<LS>()>, all_species.size()>
      move_pools{};
  std::transform(all_species.begin(), all_species.end(), move_pools.begin(),
                 move_pools.begin(),
                 [](const auto species, const auto &empty_pool) {
                   auto pool = empty_pool;
                   const auto flat = LS::data[static_cast<uint8_t>(species)];
                   std::copy_if(all_moves.begin(), all_moves.end(),
                                pool.begin(), [&flat](const auto move) {
                                  return flat[static_cast<uint8_t>(move)];
                                });
                   return pool;
                 });
  return move_pools;
}

template <typename LS> consteval auto get_n_legal_species() {
  const auto sizes = get_move_pool_sizes<LS>();
  return std::count_if(sizes.begin(), sizes.end(),
                       [](const auto n) { return n > 0; });
}

template <typename LS> consteval auto get_legal_species() {
  using PKMN::Data::all_species;
  using PKMN::Data::Species;
  const auto sizes = get_move_pool_sizes<LS>();
  std::array<Species, get_n_legal_species<LS>()> legal_species{};
  std::copy_if(all_species.begin(), all_species.end(), legal_species.begin(),
               [&sizes](const auto species) {
                 return sizes[static_cast<uint8_t>(species)] > 0;
               });
  return legal_species;
}

template <typename LS> struct FormatImpl {
  static constexpr auto MOVE_POOLS{get_move_pools<LS>()};
  static constexpr auto MOVE_POOL_SIZES{get_move_pool_sizes<LS>()};
  static constexpr const auto &move_pool(const auto species) {
    return MOVE_POOLS[static_cast<uint8_t>(species)];
  }
  static constexpr auto move_pool_size(const auto species) {
    return MOVE_POOL_SIZES[static_cast<uint8_t>(species)];
  }
  static constexpr auto max_move_pool_size{get_max_move_pool_size<LS>()};
  static constexpr auto legal_species{get_legal_species<LS>()};
};

} // namespace Format