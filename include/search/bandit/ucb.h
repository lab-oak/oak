

#pragma once

#include <search/joint.h>
#include <search/bandit/int.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace UCB {

#pragma pack(push, 1)
struct Bandit {

  struct Params {
    float c;
  };

  struct Outcome {
    float value;
    uint8_t index;
  };

  std::array<float, 9> scores;
  std::array<uint24_t, 9> visits;
  uint8_t k;

  void init(const auto k) noexcept {
    this->k = k;
    std::fill(scores.begin(), scores.begin() + k, 0);
    std::fill(visits.begin(), visits.begin() + k, 0);
  }

  bool is_init() const noexcept { return k; }

  void update(const auto &outcome) noexcept {
    scores[outcome.index] += outcome.value;
    ++visits[outcome.index];
  }

  void select(auto &device, const Params &params,
              auto &outcome) const noexcept {
    if (k == 1) {
      outcome.index = 0;
    } else {
      std::array<float, 9> q{};
      uint64_t N{};
      for (auto i = 0; i < k; ++i) {
        if (visits[i] == 0) {
          outcome.index = i;
          return;
        }
        q[i] = (float)scores[i] / visits[i];
        N += visits[i];
      }
      float log_N = std::log(N);
      float max = 0;
      const float p = 1.0 / k;
      for (auto i = 0; i < k; ++i) {
        float e = params.c * p * std::sqrt(log_N / visits[i]);
        float a = e + q[i];
        if (a > max) {
          max = a;
          outcome.index = i;
        }
      }
    }
  }
};
#pragma pack(pop)

using JointBandit = Joint<Bandit>;

}; // namespace UCB
