

#pragma once

#include <util/softmax.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>

namespace Exp3 {

constexpr float neg_inf = -std::numeric_limits<float>::infinity();

struct Params {
  Params(const float gamma) : gamma{gamma}, one_minus_gamma{1 - gamma} {}
  float gamma;
  float one_minus_gamma;
};

struct Bandit {

  std::array<float, 9> gains;
  uint8_t k;

  void init(const auto k) noexcept {
    this->k = k;
    std::fill(gains.begin(), gains.begin() + k, 0);
    std::fill(gains.begin() + k, gains.end(), neg_inf);
  }

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
      outcome.index = device.sample_pdf(policy);
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

struct JointBandit {

  struct Outcome {
    float value;
    float prob;
    uint8_t index;
  };

  struct JointOutcome {
    Outcome p1;
    Outcome p2;
  };

  Bandit p1;
  Bandit p2;

  void init(const auto m, const auto n) noexcept {
    p1.init(m);
    p2.init(n);
  }

  bool is_init() const noexcept { return p1.k != 0; }

  void select(auto &device, const Params &params,
              JointOutcome &outcome) const noexcept {
    p1.select(device, params, outcome.p1);
    p2.select(device, params, outcome.p2);
  }

  void update(const JointOutcome &outcome) noexcept {
    p1.update(outcome.p1);
    p2.update(outcome.p2);
  }
};

}; // namespace Exp3
