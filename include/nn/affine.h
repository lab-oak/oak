#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <istream>

template <bool clamp = true> class Affine {
public:
  uint32_t in_dim;
  uint32_t out_dim;

  using InputVector = Eigen::VectorXf;
  using OutputVector = Eigen::VectorXf;
  using WeightMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  WeightMatrix weights;
  OutputVector biases;

  Affine(uint32_t in_dim, uint32_t out_dim)
      : in_dim{in_dim}, out_dim{out_dim}, weights(out_dim, in_dim),
        biases(out_dim) {}

  bool read_parameters(std::istream &stream) {
    if (!stream.read(reinterpret_cast<char *>(&in_dim), sizeof(uint32_t))) {
      return false;
    }
    if (!stream.read(reinterpret_cast<char *>(&out_dim), sizeof(uint32_t))) {
      return false;
    }
    weights.resize(out_dim, in_dim);
    biases.resize(out_dim);
    if (!stream.read(reinterpret_cast<char *>(biases.data()),
                     out_dim * sizeof(float))) {
      return false;
    }
    if (!stream.read(reinterpret_cast<char *>(weights.data()),
                     out_dim * in_dim * sizeof(float))) {
      return false;
    }
    return true;
  }

  bool write_parameters(std::ostream &stream) const {
    stream.write(reinterpret_cast<const char *>(&in_dim), sizeof(uint32_t));
    stream.write(reinterpret_cast<const char *>(&out_dim), sizeof(uint32_t));
    stream.write(reinterpret_cast<const char *>(biases.data()),
                 out_dim * sizeof(float));
    stream.write(reinterpret_cast<const char *>(weights.data()),
                 out_dim * in_dim * sizeof(float));
    return !stream.fail();
  }

  void propagate(const float *input_data, float *output_data) const {
    const auto input = Eigen::Map<const InputVector>(input_data, in_dim);
    Eigen::Map<OutputVector> output(output_data, out_dim);
    output.noalias() = weights * input + biases;
    if constexpr (clamp) {
      for (std::size_t i = 0; i < out_dim; ++i) {
        output(i) = std::clamp(output(i), 0.0f, 1.0f);
      }
    }
  }

  void propagate(const uint32_t *index_data, const float *input_data,
                 float *output_data, size_t n) const {
    Eigen::Map<OutputVector> output(output_data, out_dim);
    output = biases;

    for (std::size_t k = 0; k < n; ++k) {
      uint32_t idx = index_data[k];
      float val = input_data[k];
      output.noalias() += weights.col(idx) * val;
    }
    if constexpr (clamp) {
      for (std::size_t i = 0; i < out_dim; ++i) {
        output(i) = std::clamp(output(i), 0.0f, 1.0f);
      }
    }
  }

  bool operator==(const Affine &other) const {
    return (biases == other.biases) && (weights == other.weights);
  }

  void initialize(auto &device) {
    const float k = 1.0f / std::sqrt(static_cast<float>(in_dim));
    for (std::size_t i = 0; i < out_dim; ++i) {
      biases(i) = device.uniform() * 2 * k - k;
    }
    for (std::size_t i = 0; i < out_dim; ++i) {
      for (std::size_t j = 0; j < in_dim; ++j) {
        weights(i, j) = device.uniform() * 2 * k - k;
      }
    }
  }
};