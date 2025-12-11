

#pragma once

#include <search/joint.h>
#include <search/util/int.h>
#include <search/util/softmax.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace PUCB {

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
  std::array<float, 9> priors;
  std::array<uint24_t, 9> visits;
  uint8_t k;

  void init(const auto k) noexcept {
    this->k = k;
    std::fill(scores.begin(), scores.begin() + k, 0);
    std::fill(visits.begin(), visits.begin() + k, uint24_t{});
  }

  bool is_init() const noexcept { return k; }

  void softmax_logits(const Params &, const float *logits) noexcept {
    softmax(this->priors.data(), logits, k);
  }

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
      for (auto i = k - 1; i >= 0; --i) {
        if (visits[i] == 0) {
          outcome.index = i;
          return;
        }
        q[i] = (float)scores[i] / visits[i];
        N += visits[i];
      }
      float sqrtN = std::sqrt(N);
      float max = 0;
      for (auto i = 0; i < k; ++i) {
        float e = params.c * priors[i] * sqrtN / (visits[i] + 1);
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

static_assert(sizeof(JointBandit) == 200);

}; // namespace PUCB
