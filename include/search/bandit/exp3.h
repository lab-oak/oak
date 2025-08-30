

#pragma once

#include <search/joint.h>
#include <search/util/softmax.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace Exp3 {

constexpr float neg_inf = -std::numeric_limits<float>::infinity();

struct Bandit {

  struct Outcome {
    float value;
    float prob;
    uint8_t index;
  };

  struct Params {
    Params(const float gamma) : gamma{gamma}, one_minus_gamma{1 - gamma} {}
    float gamma;
    float one_minus_gamma;
  };

  std::array<float, 9> gains;
  uint8_t k;

  void init(const auto k) noexcept {
    this->k = k;
    std::fill(gains.begin(), gains.begin() + k, 0);
    std::fill(gains.begin() + k, gains.end(), neg_inf);
  }

  bool is_init() const noexcept { return k; }

  void select(auto &device, const Params &params,
              auto &outcome) const noexcept {
    std::array<float, 9> policy;
    if (k == 1) {
      outcome.index = 0;
      outcome.prob = 1;
    } else {
      const float eta{params.gamma / k};
      softmax(policy, gains, eta);
      std::transform(policy.begin(), policy.end(), policy.begin(),
                     [eta, &params](const float value) {
                       return params.one_minus_gamma * value + eta;
                     });
      // TODO policy has eta as prob for invalid actions
      outcome.index = std::min(device.sample_pdf(policy), k - 1);
      outcome.prob = policy[outcome.index];
    }
  }

  void update(const auto &outcome) noexcept {
    if ((gains[outcome.index] += outcome.value / outcome.prob) >= 0) {
      const auto max = gains[outcome.index];
      for (auto &v : gains) {
        v -= max;
      }
    }
  }
};

using JointBandit = Joint<Bandit>;

}; // namespace Exp3
