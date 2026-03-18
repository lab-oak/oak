#pragma once

#include <nn/affine.h>

#include <fstream>
#include <vector>

namespace NN {

struct EmbeddingNet {
  Affine<Eigen::ColMajor> fc0;
  Affine<> fc1;
  std::vector<float> buf;

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

  template <Activation activation>
  void propagate(const float *input_data, float *output_data) {
    fc0.propagate<activation>(input_data, buf.data());
    fc1.propagate<activation>(buf.data(), output_data);
  }

  template <Activation activation>
  void propagate(const float *input, const auto *index, float *output_data,
                 auto n) {
    fc0.propagate<activation>(input, index, buf.data(), n);
    fc1.propagate<activation>(buf.data(), output_data);
  }

  void initialize(auto &device) {
    fc0.initialize(device);
    fc1.initialize(device);
  }
};

} // namespace NN