#pragma once

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
};