#pragma once

#include <nn/affine.h>

namespace NN {

template <bool clamp_0 = true, bool clamp_1 = true> struct EmbeddingNet {
  Affine<clamp_0> fc0;
  Affine<clamp_1> fc1;
  std::vector<float> buf;

  EmbeddingNet(uint32_t in_dim, uint32_t hidden_dim, uint32_t out_dim)
      : fc0{in_dim, hidden_dim}, fc1{hidden_dim, out_dim}, buf{} {
    buf.resize(hidden_dim);
  }

  bool operator==(const EmbeddingNet &) const = default;

  void initialize(auto &device) {
    fc0.initialize(device);
    fc1.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return fc0.read_parameters(stream) && fc1.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) const {
    return fc0.write_parameters(stream) && fc1.write_parameters(stream);
  }

  void propagate(const float *input_data, float *output_data) {
    fc0.propagate(input_data, buf.data());
    fc1.propagate(buf.data(), output_data);
  }
};

} // namespace NN