#pragma once

#include <encode/battle/policy.h>
#include <nn/affine.h>

namespace NN {

template <int in_dim, int hidden_dim, int out_dim, bool clamp_0 = true,
          bool clamp_1 = true>
struct EmbeddingNet {
  Affine<in_dim, hidden_dim, clamp_0> fc0;
  Affine<hidden_dim, out_dim, clamp_1> fc1;

  bool operator==(const EmbeddingNet &) const = default;

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
};

template <int in_dim, int hidden_dim, int value_hidden_dim,
          int policy_hidden_dim, int policy_out_dim>
struct MainNet {

  Affine<in_dim, hidden_dim> fc0;
  Affine<hidden_dim, value_hidden_dim> value_fc1;
  Affine<value_hidden_dim, 1, false> value_fc2;
  Affine<hidden_dim, policy_hidden_dim> policy1_fc1;
  Affine<policy_hidden_dim, policy_out_dim, false> policy1_fc2;
  Affine<hidden_dim, policy_hidden_dim> policy2_fc1;
  Affine<policy_hidden_dim, policy_out_dim, false> policy2_fc2;

  bool operator==(const MainNet &) const = default;

  void initialize(auto &device) {
    fc0.initialize(device);
    value_fc1.initialize(device);
    value_fc2.initialize(device);
    policy1_fc1.initialize(device);
    policy1_fc2.initialize(device);
    policy2_fc1.initialize(device);
    policy2_fc2.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return fc0.read_parameters(stream) && value_fc1.read_parameters(stream) &&
           value_fc2.read_parameters(stream) &&
           policy1_fc1.read_parameters(stream) &&
           policy1_fc2.read_parameters(stream) &&
           policy2_fc1.read_parameters(stream) &&
           policy2_fc2.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) {
    return fc0.write_parameters(stream) && value_fc1.write_parameters(stream) &&
           value_fc2.write_parameters(stream) &&
           policy1_fc1.write_parameters(stream) &&
           policy1_fc2.write_parameters(stream) &&
           policy2_fc1.write_parameters(stream) &&
           policy2_fc2.write_parameters(stream);
  }

  // TODO for now main does not do policy out, thats done by full net
  float propagate(const float *input_data) const {
    static thread_local float buffer0[hidden_dim];
    static thread_local float value_buffer1[value_hidden_dim];
    static thread_local float value_buffer2[1];
    fc0.propagate(input_data, buffer0);
    value_fc1.propagate(buffer0, value_buffer1);
    value_fc2.propagate(value_buffer1, value_buffer2);
    return 1 / (1 + std::exp(-value_buffer2[0]));
  }

  float propagate(const float *input_data, const auto m, const auto n,
                  const auto *p1_choice_index, const auto *p2_choice_index,
                  float *p1, float *p2) const {
    static thread_local float buffer0[hidden_dim];
    static thread_local float value_buffer1[value_hidden_dim];
    static thread_local float value_buffer2[1];
    static thread_local float policy1_buffer1[policy_hidden_dim];
    static thread_local float policy2_buffer1[policy_hidden_dim];

    fc0.propagate(input_data, buffer0);
    value_fc1.propagate(buffer0, value_buffer1);
    value_fc2.propagate(value_buffer1, value_buffer2);
    policy1_fc1.propagate(buffer0, policy1_buffer1);
    policy2_fc1.propagate(buffer0, policy2_buffer1);

    for (auto i = 0; i < m; ++i) {
      const auto p1_c = p1_choice_index[i];
      assert(p1_c < Encode::Policy::n_dim);
      const float logit =
          policy1_fc2.weights.row(p1_c).dot(Eigen::Map<const Eigen::VectorXf>(
              policy1_buffer1, value_hidden_dim)) +
          policy1_fc2.biases[p1_c];
      p1[i] = logit;
    }

    for (auto i = 0; i < n; ++i) {
      const auto p2_c = p2_choice_index[i];
      assert(p2_c < Encode::Policy::n_dim);
      const float logit =
          policy2_fc2.weights.row(p2_c).dot(Eigen::Map<const Eigen::VectorXf>(
              policy2_buffer1, value_hidden_dim)) +
          policy2_fc2.biases[p2_c];
      p2[i] = logit;
    }

    return 1 / (1 + std::exp(-value_buffer2[0]));
  }
};

} // namespace NN