#pragma once

#include <encode/build/trajectory.h>
#include <format/ou/data.h>
#include <nn/params.h>
#include <nn/subnet.h>
#include <train/build/trajectory.h>

namespace NN {

struct BuildNetwork {

  using T = Encode::Build::Tensorizer<Format::OU>;

  EmbeddingNet<T::n_dim, NN::Build::policy_hidden_dim, T::n_dim, true, false>
      policy_net;
  EmbeddingNet<T::n_dim, NN::Build::value_hidden_dim, T::n_dim, true, false>
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

  void propagate(const float *input_data, float *output_data) const {
    static thread_local float hidden_layer[NN::Battle::policy_hidden_dim];
    // static thread_local float value_out[1];
    policy_net.propagate(input_data, output_data);
    // value_net.propagate(input_data, value_out);
    // return value_out[0];
  }
};

}; // namespace NN
