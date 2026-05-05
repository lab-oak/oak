#pragma once

#include <search/mcts.h>

inline std::tuple<double, double, double>
expl(const MCTS::Output &a, const MCTS::Output &o,
     const RuntimePolicy::Options &p) {
  auto p1_policy = RuntimePolicy::get_policy(o.p1, p);
  auto p2_policy = RuntimePolicy::get_policy(o.p2, p);
  std::array<double, 9> p1_rewards{};
  std::array<double, 9> p2_rewards{};
  for (auto i = 0; i < a.p1.k; ++i) {
    for (auto j = 0; j < a.p2.k; ++j) {
      auto value = a.visit_matrix[i][j] == 0
                       ? .5
                       : a.value_matrix[i][j] / a.visit_matrix[i][j];
      p1_rewards[i] += p2_policy[j] * value;
      p2_rewards[j] += p1_policy[i] * value;
    }
  }
  auto min = *std::min_element(p2_rewards.begin(), p2_rewards.begin() + a.p2.k);
  auto max = *std::max_element(p1_rewards.begin(), p1_rewards.begin() + a.p1.k);
  double u = 0;
  for (auto i = 0; i < a.p1.k; ++i) {
    u += p1_policy[i] * p1_rewards[i];
  }
  return {min, u, max};
}
