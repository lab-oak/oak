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

  using InputVector = Eigen::Matrix<float, kIn, 1>;
  using OutputVector = Eigen::Matrix<float, kOut, 1>;

  using WeightMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  WeightMatrix weights;
  OutputVector biases;

  Affine() : weights(kOut, kIn) {}

  bool read_parameters(std::istream &stream) {
    if (!stream.read(reinterpret_cast<char *>(biases.data()),
                     kOut * sizeof(float)))
      return false;
    if (!stream.read(reinterpret_cast<char *>(weights.data()),
                     kOut * kIn * sizeof(float)))
      return false;
    return true;
  }

  void propagate(const float *input_data, float *output_data) const {
    const InputVector input = Eigen::Map<const InputVector>(input_data);
    const OutputVector output = weights * input + biases;

    if constexpr (clamp) {
      for (std::size_t i = 0; i < kOut; ++i)
        output_data[i] = std::clamp(output(i), 0.0f, 1.0f);
    } else {
      Eigen::Map<OutputVector>(output_data).noalias() = output;
    }
  }
};
