#pragma once

#include <encode/build/compressed-trajectory.h>
#include <format/ou/data.h>
#include <nn/default-hyperparameters.h>
#include <nn/embedding-net.h>
#include <train/build/trajectory.h>

namespace NN {

namespace Build {

struct Network {

  // using T = Encode::Build::Tensorizer<Format::OU>;

  TeamBuildingNet policy_net;
  TeamBuildingNet value_net;

  void initialize(auto &device) {
    policy_net.initialize(device);
    value_net.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return policy_net.read_parameters(stream) &&
           value_net.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) const {
    return policy_net.write_parameters(stream) &&
           value_net.write_parameters(stream);
  }

  void inference(const float *input, float *output) {
    policy_net.propagate<Activation::relu, Activation::relu, Activation::none>(
        input, output);
  }
};

} // namespace Build

}; // namespace NN
