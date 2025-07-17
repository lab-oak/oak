#pragma once

#include <data/legal-moves.h>
#include <data/options.h>

namespace Data {
consteval auto get_move_pools() {
  using enum Data::Move;
  auto MOVE_POOLS = Data::LEARNSETS;
  for (auto &move_pool : MOVE_POOLS) {
    move_pool[static_cast<uint8_t>(Dig)] = false;
    move_pool[static_cast<uint8_t>(Fly)] = false;
    move_pool[static_cast<uint8_t>(Fissure)] = false;
    move_pool[static_cast<uint8_t>(HornDrill)] = false;
    move_pool[static_cast<uint8_t>(Minimize)] = false;
    move_pool[static_cast<uint8_t>(DoubleTeam)] = false;
  }
  return MOVE_POOLS;
}

constexpr auto MOVE_POOLS = get_move_pools();

consteval auto get_move_pool_sizes() {
  std::array<uint8_t, 152> move_pool_sizes{};
  for (auto species = 0; species <= 151; ++species) {
    auto size = 0;
    for (auto m : MOVE_POOLS[species]) {
      size += m;
    }
    move_pool_sizes[species] = size;
  }
  return move_pool_sizes;
}

constexpr auto MOVE_POOL_SIZES = get_move_pool_sizes();

constexpr auto move_pool_size(const auto species) noexcept {
  return MOVE_POOL_SIZES[static_cast<uint8_t>(species)];
}

constexpr auto max_move_pool_size =
    *std::max_element(MOVE_POOL_SIZES.begin(), MOVE_POOL_SIZES.end());

consteval auto get_move_pools_flat() {
  std::array<std::array<Data::Move, max_move_pool_size>, 152> list{};
  for (auto i = 0; i <= 151; ++i) {
    auto index = 0;
    for (uint8_t m = 0; m < 166; ++m) {
      if (MOVE_POOLS[i][m]) {
        list[i][index++] = static_cast<Data::Move>(m);
      }
    }
  }
  return list;
}

constexpr auto MOVE_POOLS_FLAT = get_move_pools_flat();

constexpr const auto &move_pool(const auto species) noexcept {
  return MOVE_POOLS_FLAT[static_cast<uint8_t>(species)];
}
} // namespace Data