#pragma once

#include <encode/build/compressed-trajectory.h>
#include <format/ou/data.h>
#include <nn/default-hyperparameters.h>
#include <nn/ffn.h>
#include <train/build/trajectory.h>

namespace NN {

namespace Build {

struct Network {

  TeamBuildingNet policy_net;
  // unused
  TeamBuildingNet value_net;

  void initialize(auto &device) {
    policy_net.initialize(device);
    value_net.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return policy_net.read_parameters(stream) &&
           value_net.read_parameters(stream);
  }

  void inference(const float *input, float *output) {
    policy_net.propagate<Activation::relu, Activation::relu, Activation::none>(
        input, output);
  }
};

} // namespace Build

}; // namespace NN
