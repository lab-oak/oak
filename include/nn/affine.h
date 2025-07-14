#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <istream>

template <std::size_t in_dim, std::size_t out_dim, bool clamp = true>
class Affine {
public:
  static constexpr std::size_t kIn = in_dim;
  static constexpr std::size_t kOut = out_dim;

  using InputVector = Eigen::VectorXf;
  using OutputVector = Eigen::VectorXf;

  using WeightMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  WeightMatrix weights;
  OutputVector biases;

  Affine() : weights(kOut, kIn), biases(kOut) {}

  void initialize(auto &device) {
    const float k = 1.0f / std::sqrt(static_cast<float>(kIn));
    for (std::size_t i = 0; i < kOut; ++i) {
      biases(i) = device.uniform() * 2 * k - k;
    }
    for (std::size_t i = 0; i < kOut; ++i) {
      for (std::size_t j = 0; j < kIn; ++j) {
        weights(i, j) = device.uniform() * 2 * k - k;
      }
    }
  }

  bool read_parameters(std::istream &stream) {
    if (!stream.read(reinterpret_cast<char *>(biases.data()),
                     kOut * sizeof(float))) {
      return false;
    }
    if (!stream.read(reinterpret_cast<char *>(weights.data()),
                     kOut * kIn * sizeof(float))) {
      return false;
    }
    return true;
  }

  bool write_parameters(std::ostream &stream) const {
    stream.write(reinterpret_cast<const char *>(biases.data()),
                 kOut * sizeof(float));
    stream.write(reinterpret_cast<const char *>(weights.data()),
                 kOut * kIn * sizeof(float));
    return !stream.fail();
  }

  void propagate(const float *input_data, float *output_data) const {
    const auto input = Eigen::Map<const InputVector>(input_data, kIn);
    Eigen::Map<OutputVector> output(output_data, kOut);
    output.noalias() = weights * input + biases;
    if constexpr (clamp) {
      for (std::size_t i = 0; i < kOut; ++i) {
        output(i) = std::clamp(output(i), 0.0f, 1.0f);
      }
    }
  }
};