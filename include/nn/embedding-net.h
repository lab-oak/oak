#pragma once

#include <nn/affine.h>

#include <fstream>
#include <vector>

namespace NN {

template <bool clamp_0 = true, bool clamp_1 = true> struct EmbeddingNet {
  Affine<clamp_0> fc0;
  Affine<clamp_1> fc1;
  std::vector<float> buf;

  void initialize(auto &device) {
    fc0.initialize(device);
    fc1.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    const bool ok = fc0.read_parameters(stream) && fc1.read_parameters(stream);
    if (!ok) {
      return false;
    }
    buf.resize(fc0.out_dim);
    return true;
  }

  bool write_parameters(std::ostream &stream) const {
    return fc0.write_parameters(stream) && fc1.write_parameters(stream);
  }

  void propagate(const float *input_data, float *output_data) {
    fc0.propagate(input_data, buf.data());
    fc1.propagate(buf.data(), output_data);
  }

  void propagate(const float *input, const auto *index, float *output_data,
                 auto n) {
    fc0.propagate(input, index, buf.data(), n);
    fc1.propagate(buf.data(), output_data);
  }
};

} // namespace NN