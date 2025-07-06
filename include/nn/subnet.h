#pragma once

#include <nn/affine.h>

namespace NN {

template <int in_dim, int hidden_dim, int out_dim, bool clamp_0 = true,
          bool clamp_1 = true>
struct EmbeddingNet {
  Affine<in_dim, hidden_dim, clamp_0> fc0;
  Affine<hidden_dim, out_dim, clamp_1> fc1;

  bool read_parameters(std::istream &stream) {
    return fc0.read_parameters(stream) && fc1.read_parameters(stream);
  }

  void propagate(const float *input_data, float *output_data) const {
    static thread_local float hidden_layer[hidden_dim];
    fc0.propagate(input_data, hidden_layer);
    fc1.propagate(hidden_layer, output_data);
  }
};

constexpr auto pokemon_out_dim = 39;
constexpr auto active_out_dim = 55;

struct MainNet {
  static constexpr int in_dim = 512;
  static constexpr int hidden_dim = 32;
  static constexpr int out_dim = 1;

  Affine<in_dim, hidden_dim> fc0;
  Affine<hidden_dim, hidden_dim> fc1;
  Affine<hidden_dim, out_dim, false> fc2;

  bool read_parameters(std::istream &stream) {
    return fc0.read_parameters(stream) && fc1.read_parameters(stream) &&
           fc2.read_parameters(stream);
  }

  float propagate(const float *input_data, float *output_data) const {
    static thread_local float b0[hidden_dim];
    static thread_local float b1[hidden_dim];
    static thread_local float b2[out_dim];
    fc0.propagate(input_data, b0);
    fc1.propagate(b0, b1);
    fc2.propagate(b1, b2);
    return 1 / (1 + std::exp(-b2[0]));
  }
};

} // namespace NN