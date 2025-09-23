#pragma once

#include <encode/build/trajectory.h>
#include <nn/build/network.h>
#include <train/build/trajectory.h>

// Rollout battle network

namespace TeamBuilding {

// Expects a full 6. If truncate, then also shuffle
struct Omitter {

  int max_pokemon{6};
  double pokemon_delete_prob{};
  double move_delete_prob{};

  void shuffle_and_truncate(auto &device, auto &team) const {
    if (max_pokemon < team.size()) {
      std::mt19937 rd{device.uniform_64()};
      std::shuffle(team.begin(), team.end(), rd);
    }
    team.resize(max_pokemon);
  }

  auto delete_info(auto &device, auto &team) const {
    for (auto &set : team) {
      if (device.uniform() < pokemon_delete_prob) {
        set = PKMN::Set{};
      } else {
        for (auto &move : set.moves) {
          if (device.uniform() < move_delete_prob) {
            move = PKMN::Data::Move::None;
          }
        }
      }
    }
  }
};

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

  auto input = Encode::Build::Tensorizer<>::write(team);
  std::array<float, Encode::Build::Tensorizer<>::n_dim> logits;
  auto actions = Encode::Build::Actions<>::get_singleton_additions(team);

  const auto go = [&]() {
    // get action indices
    std::vector<int> indices;
    std::transform(actions.begin(), actions.end(), std::back_inserter(indices),
                   [](auto action) {
                     return Encode::Build::Tensorizer<>::action_index(action);
                   });

    network.propagate(input.data(), logits.data());

    // get legal logits, softmax, sample action, apply
    std::vector<float> legal_logits;
    std::transform(indices.begin(), indices.end(),
                   std::back_inserter(legal_logits),
                   [&logits](const auto index) { return logits[index]; });
    const auto policy = softmax(legal_logits);
    const auto index = device.sample_pdf(policy);
    const auto action = actions[index];
    apply_action(trajectory.terminal, action);
    input[index] = 1.0;

    trajectory.updates.emplace_back(
        Trajectory::Update{actions, index, policy[index]});
  };

  while (!actions.empty()) {
    go();
    actions =
        Encode::Build::Actions<>::get_singleton_additions(trajectory.terminal);
  }
  actions = Encode::Build::Actions<>::get_lead_swaps(trajectory.terminal);
  if (!actions.empty()) {
    go();
  }

  return trajectory;
}

} // namespace TeamBuilding