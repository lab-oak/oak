#pragma once

#include <format/OU/legal-moves.h>
#include <libpkmn/data/moves.h>

namespace Format {

namespace OU {

constexpr auto MOVE_POOL_SIZES = get_move_pool_sizes<LEARNSETS>();
constexpr auto MOVE_POOLS_FLAT = get_move_pools_flat<LEARNSETS>();

constexpr auto max_move_pool_size = get_max_move_pool_size<LEARNSETS>();
using MovePool = std::array<PKMN::Data::Move, max_move_pool_size>;
constexpr auto move_pool_size(const auto species) {
  return MOVE_POOL_SIZES[static_cast<uint8_t>(species)];
}
constexpr const auto &move_pool(const auto species) noexcept {
  return MOVE_POOLS_FLAT[static_cast<uint8_t>(species)];
}

} // namespace OU

} // namespace Format