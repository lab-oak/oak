#pragma once

#include <pkmn.h>

#include <assert.h>

#include <array>
#include <map>
#include <memory>

namespace Tree {

template <typename BanditData, typename Obs> struct Node {
  using Key = std::tuple<uint8_t, uint8_t, Obs>;
  using Value = Node<BanditData, Obs>;
  using ChanceMap = std::map<Key, Value>;
  BanditData _data;
  ChanceMap _map;

  const auto &stats() const noexcept { return _data; }
  auto &stats() noexcept { return _data; }

  void init(auto p1_choices, auto p2_choices) noexcept {
    _data.init(p1_choices, p2_choices);
  }

  bool is_init() const noexcept { return _data.is_init(); }

  Node *operator()(auto p1_index, auto p2_index, auto obs) {
    auto &node = _map[{p1_index, p2_index, obs}];
    return &node;
  }

  auto get(auto p1_index, auto p2_index, auto obs) {
    return _map.find({p1_index, p2_index, obs});
  }

  // Node *operator[](auto p1_index, auto p2_index, auto obs) {
  //   return _map[{p1_index, p2_index, obs}].get();
  // }

  std::unique_ptr<Node> release_child(auto p1_index, auto p2_index, auto obs) {
    return std::move(_map[{p1_index, p2_index, obs}]);
  }
};

template <typename BanditData, typename Obs, size_t size,
          typename index_type = uint32_t>
struct Table {
  std::array<BanditData, size> stats_table;
  using Key = std::tuple<index_type, uint8_t, uint8_t, Obs>;
  std::map<uint64_t, index_type> index_map;
  index_type index;

  const auto &stats(index_type idx) const noexcept { return stats_table[idx]; }
  auto &stats(index_type idx) noexcept { return stats_table[idx]; }

  // index_type get(index_type index, uint8_t i, uint8_t j,
  //                const Obs &obs) noexcept {
  //   index_type &child_index = index_map[Key{index, i, j, obs}];
  //   if (child_index == 0) {
  //     child_index = ++index;
  //   }
  //   return child_index;
  // }
  index_type get(uint64_t key) noexcept {
    index_type &child_index = index_map[key];
    if (child_index == 0) {
      child_index = ++index;
    }
    return child_index;
  }

  void reset() noexcept { index = 0; }
};

}; // namespace Tree