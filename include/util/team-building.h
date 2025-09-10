#pragma once

#include <encode/build/trajectory.h>
#include <nn/build/network.h>
#include <train/build/trajectory.h>

// Rollout battle network

namespace TeamBuilding {

const auto team_string = [](const auto &team) {
  std::stringstream ss{};
  for (const auto &set : team) {
    ss << PKMN::species_string(set.species) << ": ";
    for (const auto moveid : set.moves) {
      ss << PKMN::move_string(moveid) << ' ';
    }
    ss << '\n';
  }
  return ss.str();
};

auto softmax(auto &x) {
  auto y = x;
  std::transform(y.begin(), y.end(), y.begin(),
                 [](const auto v) { return std::exp(v); });
  const auto sum = std::accumulate(y.begin(), y.end(), 0.0);
  std::transform(y.begin(), y.end(), y.begin(),
                 [sum](const auto v) { return v / sum; });
  return y;
}

[[nodiscard]] auto rollout_build_network(auto &device, auto &network,
                                         const auto &team) {
  using namespace Train::Build;

  Trajectory trajectory{};
  trajectory.initial = team;
  trajectory.terminal = team;

  const auto initial_accumulate = [&network](const float *input_data,
                                             float *output_data) {
    using T = std::remove_cvref_t<decltype(network.policy_net.fc0)>;
    const auto input = Eigen::Map<const Eigen::VectorXf>(input_data, T::kIn);
    Eigen::Map<Eigen::VectorXf> output(output_data, T::kOut);
    output.noalias() =
        network.policy_net.fc0.weights * input + network.policy_net.fc0.biases;
  };

  auto input = Encode::Build::Tensorizer<>::write(team);
  std::array<float, NN::Build::policy_hidden_dim> accumulate;
  initial_accumulate(input.data(), accumulate.data());
  std::array<float, NN::Build::policy_hidden_dim> activations;

  std::array<float, Encode::Build::Tensorizer<>::n_dim> logits;
  auto actions = Encode::Build::Actions<>::get_singleton_additions(team);

  const auto go = [&]() {
    // get action indices
    std::vector<int> indices;
    std::transform(actions.begin(), actions.end(), std::back_inserter(indices),
                   [](auto action) {
                     return Encode::Build::Tensorizer<>::action_index(action);
                   });

    // network.propagate(input.data(), logits.data());
    for (auto i = 0; i < accumulate.size(); ++i) {
      activations[i] = std::clamp(accumulate[i], 0.0f, 1.0f);
    }
    // network.policy_net.fc1.propagate(activations.data(), logits.data());
    // get legal logits, softmax, sample action, apply
    std::vector<float> legal_logits;
    // legal_logits.reserve(indices.size());
    // std::transform(indices.begin(), indices.end(),
    //                std::back_inserter(legal_logits),
    //                [&logits](const auto index) { return logits[index]; });

    for (const auto idx : indices) {
      float logit = network.policy_net.fc1.biases[idx];
      for (int h = 0; h < activations.size(); ++h) {
        logit += activations[h] * network.policy_net.fc1.weights(idx, h);
      }
      legal_logits.push_back(logit);
    }
    const auto policy = softmax(legal_logits);
    const auto index = device.sample_pdf(policy);
    const auto action = actions[index];
    apply_action(trajectory.terminal, action);
    // input[index] = 1.0;
    for (int h = 0; h < accumulate.size(); ++h) {
      accumulate[h] += network.policy_net.fc0.weights(h, index); // if weights is [H × input_dim]
    }
    trajectory.updates.emplace_back(
        Trajectory::Update{actions, index, policy[index]});
  };

  while (!actions.empty()) {
    go();
    actions =
        Encode::Build::Actions<>::get_singleton_additions(trajectory.terminal);
  }
  actions = Encode::Build::Actions<>::get_lead_actions(trajectory.terminal);
  if (!actions.empty()) {
    go();
  }

  return trajectory;
}

} // namespace TeamBuilding