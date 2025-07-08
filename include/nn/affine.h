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

  Affine() : weights(kOut, kIn) {
    const float k = 1.0f / std::sqrt(static_cast<float>(kIn));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-k, k);

    for (std::size_t i = 0; i < kOut; ++i)
      biases(i) = dist(gen);

    for (std::size_t i = 0; i < kOut; ++i)
      for (std::size_t j = 0; j < kIn; ++j)
        weights(i, j) = dist(gen);
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
