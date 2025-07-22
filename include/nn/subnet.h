#pragma once

#include <nn/affine.h>

namespace NN {

template <int in_dim, int hidden_dim, int out_dim, bool clamp_0 = true,
          bool clamp_1 = true>
struct EmbeddingNet {
  Affine<in_dim, hidden_dim, clamp_0> fc0;
  Affine<hidden_dim, out_dim, clamp_1> fc1;

  void initialize(auto &device) {
    fc0.initialize(device);
    fc1.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return fc0.read_parameters(stream) && fc1.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) {
    return fc0.write_parameters(stream) && fc1.write_parameters(stream);
  }

  void propagate(const float *input_data, float *output_data) const {
    static thread_local float hidden_layer[hidden_dim];
    fc0.propagate(input_data, hidden_layer);
    fc1.propagate(hidden_layer, output_data);
  }

  bool operator==(const EmbeddingNet &) const = default;
};

template <int hidden_dim, int policy_out_dim> struct MainNet {
  static constexpr int in_dim = 512;

  Affine<in_dim, hidden_dim> fc0;
  Affine<hidden_dim, hidden_dim> value_fc1;
  Affine<hidden_dim, 1, false> value_fc2;
  Affine<hidden_dim, hidden_dim> policy_fc1;
  Affine<hidden_dim, policy_out_dim, false> policy_fc2;

  void initialize(auto &device) {
    fc0.initialize(device);
    value_fc1.initialize(device);
    value_fc2.initialize(device);
    policy_fc1.initialize(device);
    policy_fc2.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return fc0.read_parameters(stream) && value_fc1.read_parameters(stream) &&
           value_fc2.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) {
    return fc0.write_parameters(stream) && value_fc1.write_parameters(stream) &&
           value_fc2.write_parameters(stream);
  }

  float propagate(const float *input_data, float *policy_output) const {
    static thread_local float b0[hidden_dim];
    static thread_local float value_b1[hidden_dim];
    static thread_local float value_b2[1];
    static thread_local float policy_b1[hidden_dim];
    fc0.propagate(input_data, b0);
    value_fc1.propagate(b0, value_b1);
    value_fc2.propagate(value_b1, value_b2);
    if (policy_output) {
      policy_fc1.propagate(b0, policy_b1);
      policy_fc2.propagate(policy_b1, policy_output);
    }
    return 1 / (1 + std::exp(-value_b2[0]));
  }

  bool operator==(const MainNet &) const = default;
};

} // namespace NN