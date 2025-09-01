#pragma once

/*

Sytactic sugar over the raw tables. Also improves performance w.r.t. scanning (e.g. writing TrjaectoryInput)
since the movesets are sparse.

*/

namespace Format {

using LearnsetTable = std::array<std::array<bool, 166>, 151>;

template <LearnsetTable ls> consteval auto get_move_pool_sizes() {
  std::array<uint8_t, 152> move_pool_sizes{};
  for (auto species = 0; species <= 151; ++species) {
    auto size = 0;
    for (auto m : ls[species]) {
      size += m;
    }
    move_pool_sizes[species] = size;
  }
  return move_pool_sizes;
}

template <LearnsetTable ls>
consteval auto get_max_move_pool_size() {
  constexpr auto size = get_move_pool_sizes<ls>();
  constexpr auto max = *std::max_element(size.begin(), size.end());
  return max;
}

template <LearnsetTable ls> consteval auto get_move_pools_flat() {

  constexpr max = get_max_move_pool_size<ls>();
  using MovePool = std::array<PKMN::Data::Move, max>;
  std::array<MovePool, 152> list{};
  for (auto i = 0; i < 152; ++i) {
    auto index = 0;
    for (uint8_t m = 0; m < 166; ++m) {
      if (MOVE_POOLS[i][m]) {
        list[i][index++] = static_cast<PKMN::Data::Move>(m);
      }
    }
  }
  return list;
}

} // namespace Format