#pragma once

#include <encode/build/trajectory.h>
#include <nn/build/network.h>
#include <train/build/trajectory.h>

// Rollout battle network

namespace TeamBuilding {

auto softmax(auto &x) {
  auto y = x;
  std::transform(y.begin(), y.end(), y.begin(), [](const auto v){return std::exp(v);});
  const auto sum = std::accumulate(y.begin(), y.end(), 0.0);
  std::transform(y.begin(), y.end(), y.begin(), [sum](const auto v){return v / sum;});
  return y;
}

[[nodiscard]] auto rollout_build_network(auto &device, auto &network,
                                          const auto &team) {
  using namespace Train::Build;

  Trajectory trajectory{};
  trajectory.initial = team;

  auto input = Encode::Build::OUFormatter::write(team);
  std::array<float, Encode::Build::OUFormatter::n_dim> logits;

  auto actions = Encode::Build::OUFormatter::get_singleton_additions(team);

  while (!actions.empty()) {

    // get action indices
    std::vector<int> indices;
    std::transform(
        actions.begin(), actions.end(), std::back_inserter(indices),
        [](auto action) { return Encode::Build::OUFormatter::action_index(action); });

    network.propagate(input.data(), logits.data());

    // get legal logits, softmax, sample action, apply
    std::vector<float> legal_logits;
    std::transform(indices.begin(), indices.end(),
                   std::back_inserter(legal_logits),
                   [&logits](const auto index) { return logits[index]; });
    const auto policy = softmax(legal_logits);
    const auto index = device.sample_pdf(policy);
    const auto action = actions[index];
    apply_action(team, action);

    trajectory.updates.emplace_back(Trajectory::Update{actions, index, policy[index]});

    actions = Encode::Build::OUFormatter::get_singleton_additions(team);
  }

  // actions = Encode::Build::OUFormatter::get_switch_actions();

  trajectory.terminal = team;

  return trajectory;
}

}