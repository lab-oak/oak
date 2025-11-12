#pragma once

#include <encode/build/trajectory.h>
#include <format/ou/data.h>
#include <nn/default-hyperparameters.h>
#include <nn/embedding-net.h>
#include <train/build/trajectory.h>

namespace NN {

namespace Build {

struct Network {

  using T = Encode::Build::Tensorizer<Format::OU>;

  EmbeddingNet<true, false> policy_net;
  EmbeddingNet<true, false> value_net;

  Network()
      : policy_net{T::n_dim, NN::Build::Default::policy_hidden_dim, T::n_dim},
        value_net{T::n_dim, NN::Build::Default::value_hidden_dim, 1} {}

  void initialize(auto &device) {
    policy_net.initialize(device);
    value_net.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return policy_net.read_parameters(stream) &&
           value_net.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) const {
    return policy_net.write_parameters(stream) &&
           value_net.write_parameters(stream);
  }

  void propagate(const float *input_data, float *output_data) {
    policy_net.propagate(input_data, output_data);
  }
};

} // namespace Build

}; // namespace NN
