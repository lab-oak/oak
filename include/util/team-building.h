#pragma once

#include <encode/team.h>
#include <format/OU/move-pools.h>
#include <nn/build-network.h>
#include <train/build-trajectory>

// Rollout battle network

auto softmax(auto &x) {
  auto y = x;
  return y;
}

[[nodiscard]] auto rollout_without_reward(auto &device, auto &network,
                                          const auto &team) {
  using namespace Train::TeamBuilding;

  Trajectory trajectory{};
  trajectory.initial = team;

  auto input = Encode::Team::write(team);
  std::array<float, Encode::Team::n_dim> logits;

  auto actions = Encode::Team::get_fill_actions(team);

  while (!actions.empty()) {

    // get action indices
    std::vector<int> indices;
    std::transform(
        actions.begin(), actions.end(), std::back_inserter(indices),
        [](auto action) { return Encode::Team::action_index(action); });

    // ignore value output, only used in learning
    const auto _ = network.propagate(input.data(), logits.data());

    // get legal logits, softmax, sample action, apply
    std::vector<float> legal_logits;
    std::transform(indices.begin(), indices.end(),
                   std::back_inserter(legal_logits),
                   [&logits](int index) { return logits[i]; });
    const auto policy = softmax(legal_logits);
    const auto index = device.sample_pdf(policy);
    const auto acion = actions[index];
    apply_action(team, action);

    trajectory.updates.emplace_back({actions, index, policy[index]});

    actions = Encode::Team::get_fill_actions(team);
  }

  actions = Encode::Team::get_switch_actions();

  trajectory.terminal = team;

  return trajectory;
}