#pragma once

#include <encode/team.h>
#include <nn/params.h>
#include <nn/subnet.h>
#include <train/build-trajectory.h>

namespace NN {

struct BuildNetwork {

  EmbeddingNet<Encode::Team::in_dim, builder_policy_hidden_dim,
               Encode::Team::out_dim, true, false>
      policy_net;
  EmbeddingNet<Encode::Team::in_dim, builder_value_hidden_dim,
               Encode::Team::out_dim, true, false>
      value_net;

  void initialize(auto &device) {
    policy_net.initialize(device);
    value_net.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return policy_net.read_parameters(stream) &&
           value_net.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) {
    return policy_net.write_parameters(stream) &&
           value_net.write_parameters(stream);
  }

  float propagate(const float *input_data, float *output_data) const {
    static thread_local float hidden_layer[hidden_dim];
    static thread_local float value_out[1];
    policy_net.propagate(input_data, output_data);
    value_net.propagate(input_data, value_out);
    return value_out[0];
  }
};

auto rollout_build_network(auto &team, NN::BuildNetwork &build_net,
                           auto &device) -> Train::BuildTrajectory {

  using namespace Encode::Team;

  auto [traj, i] = Encode::Team::initial_trajectory(team);

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
    assert(sum > 0);
    for (auto &x : output) {
      x /= sum;
    }

    const auto index = device.sample_pdf(output);
    traj.frames[i++] = Train::EncodedBuildTrajectory2::Update{
        static_cast<uint16_t>(index), output[index]};
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
