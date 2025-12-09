#pragma once

#include <encode/battle/policy.h>
#include <nn/affine.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

namespace NN::Battle {

inline constexpr float sigmoid(const float x) { return 1 / (1 + std::exp(-x)); }

struct MainNet {

  Affine<> fc0;
  Affine<> value_fc1;
  Affine<false> value_fc2;
  Affine<> p1_policy_fc1;
  Affine<false> p1_policy_fc2;
  Affine<> p2_policy_fc1;
  Affine<false> p2_policy_fc2;

  std::vector<float> buffer;
  std::vector<float> value_buffer;
  std::vector<float> p1_policy_buffer;
  std::vector<float> p2_policy_buffer;

  bool read_parameters(std::istream &stream) {
    const bool ok = fc0.read_parameters(stream) &&
                    value_fc1.read_parameters(stream) &&
                    value_fc2.read_parameters(stream) &&
                    p1_policy_fc1.read_parameters(stream) &&
                    p1_policy_fc2.read_parameters(stream) &&
                    p2_policy_fc1.read_parameters(stream) &&
                    p2_policy_fc2.read_parameters(stream);

    if (!ok) {
      return false;
    }
    buffer.resize(fc0.out_dim);
    value_buffer.resize(value_fc1.out_dim);
    p1_policy_buffer.resize(p1_policy_fc1.out_dim);
    p2_policy_buffer.resize(p2_policy_fc1.out_dim);
    return true;
  }

  bool write_parameters(std::ostream &stream) const {
    return fc0.write_parameters(stream) && value_fc1.write_parameters(stream) &&
           value_fc2.write_parameters(stream) &&
           p1_policy_fc1.write_parameters(stream) &&
           p1_policy_fc2.write_parameters(stream) &&
           p2_policy_fc1.write_parameters(stream) &&
           p2_policy_fc2.write_parameters(stream);
  }

  float propagate(const float *input_data) {
    float output;
    fc0.propagate(input_data, buffer.data());
    value_fc1.propagate(buffer.data(), value_buffer.data());
    value_fc2.propagate(value_buffer.data(), &output);
    return sigmoid(output);
  }

  template <bool use_value = true>
  auto propagate(const float *input_data, const auto m, const auto n,
                 const auto *p1_choice_index, const auto *p2_choice_index,
                 float *p1, float *p2)
      -> std::conditional_t<use_value, float, void> {
    float output;
    fc0.propagate(input_data, buffer.data());
    if constexpr (use_value) {
      value_fc1.propagate(buffer.data(), value_buffer.data());
      value_fc2.propagate(value_buffer.data(), &output);
    }
    p1_policy_fc1.propagate(buffer.data(), p1_policy_buffer.data());
    p2_policy_fc1.propagate(buffer.data(), p2_policy_buffer.data());

    for (auto i = 0; i < m; ++i) {
      const auto p1_c = p1_choice_index[i];
      assert(p1_c < Encode::Battle::Policy::n_dim);
      const float logit =
          p1_policy_fc2.weights.row(p1_c).dot(Eigen::Map<const Eigen::VectorXf>(
              p1_policy_buffer.data(), p1_policy_fc1.out_dim)) +
          p1_policy_fc2.biases[p1_c];
      assert(!std::isnan(logit));
      p1[i] = logit;
    }

    for (auto i = 0; i < n; ++i) {
      const auto p2_c = p2_choice_index[i];
      assert(p2_c < Encode::Battle::Policy::n_dim);
      const float logit =
          p2_policy_fc2.weights.row(p2_c).dot(Eigen::Map<const Eigen::VectorXf>(
              p2_policy_buffer.data(), p2_policy_fc1.out_dim)) +
          p2_policy_fc2.biases[p2_c];
      assert(!std::isnan(logit));
      p2[i] = logit;
    }
    if constexpr (use_value) {
      return sigmoid(output);
    }
  }
};

} // namespace NN::Battle