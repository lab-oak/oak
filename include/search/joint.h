#pragma once

#include <utility>

#pragma pack(push, 1)
template <typename Bandit> struct Joint {
  using Params = typename Bandit::Params;
  using Outcome = typename Bandit::Outcome;

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

  bool is_init() const noexcept { return p1.is_init(); }

  void select(auto &device, const Params &params,
              JointOutcome &outcome) const noexcept {
    p1.select(device, params, outcome.p1);
    p2.select(device, params, outcome.p2);
  }

  void update(const JointOutcome &outcome) noexcept {
    p1.update(outcome.p1);
    p2.update(outcome.p2);
  }

  void init_priors(const float *p1_priors, const float *p2_priors) noexcept
    requires requires(const float *ptr) {
      std::declval<Bandit>().init_priors(ptr);
    }
  {
    p1.init_priors(p1_priors);
    p2.init_priors(p2_priors);
  }
};
#pragma pack(pop)