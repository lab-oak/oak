#pragma once

#include <map>
#include <utility>

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

  Node &operator()(auto p1_index, auto p2_index, auto obs) {
    return _map[{p1_index, p2_index, obs}];
  }

  auto find(auto p1_index, auto p2_index, auto obs) {
    return _map.find({p1_index, p2_index, obs});
  }
};

size_t count(const auto &node) {
  size_t c = 1;
  for (const auto &pair : node._map) {
    c += count(pair.second);
  }
  return c;
}

}; // namespace Tree