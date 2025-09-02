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

} // namespace NN