#pragma once

#include <encode/team.h>
#include <nn/params.h>
#include <nn/subnet.h>
#include <train/build-trajectory.h>
namespace NN {

using BuildNetwork = EmbeddingNet<Encode::Team::in_dim, team_hidden_dim,
                                  Encode::Team::out_dim, true, false>;

auto rollout_build_network(auto &team, NN::BuildNetwork &build_net,
                           auto &device) -> Train::BuildTrajectory {

  using namespace Encode::Team;

  auto traj = Encode::Team::initial_trajectory(team);

  std::array<float, in_dim> input{};
  Encode::Team::write(team, input.data());
  std::array<float, out_dim> mask;
  std::array<float, out_dim> output;

  while (true) {
    mask = {};

    const bool only_requires_lead_selection =
        Encode::Team::write_policy_mask(team, mask.data());

    build_net.propagate(input.data(), output.data());

    // softmax
    float sum = 0;
    for (auto k = 0; k < out_dim; ++k) {
      if (mask[k]) {
        output[k] = std::exp(output[k]);
        sum += output[k];
      } else {
        output[k] = 0;
      }
    }
    for (auto &x : output) {
      x /= sum;
    }

    const auto index = device.sample_pdf(output);
    traj.frames[i++] =
        Train::ActionPolicy{static_cast<uint16_t>(index), output[index]};
    input[index] = 1;
    const auto [s, m] = species_move_list(index);
    apply_index_to_team(team, s, m);

    if (only_requires_lead_selection) {
      break;
    }
  }

  return traj;
}

}; // namespace NN
