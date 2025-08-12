#pragma once

namespace RuntimePolicy {

struct Options {
  char mode = 'n';
  double temp = 1;
  double min_prob = 0;
  double nash_weight = .5;

  std::string to_string() const {
    std::stringstream ss{};
    ss << " mode: " << mode;
    ss << "temp: " << temp;
    ss << " min-prob: " << min_prob;
    return ss.str();
  }
};

int process_and_sample(auto &device, const auto &empirical, const auto &nash,
                       const auto &policy_options) {
  const double t = policy_options.temp;
  if (policy_options.temp <= 0) {
    throw std::runtime_error("Use positive policy power");
  }

  std::array<double, 9> policy{};
  if (policy_options.mode == 'e') {
    policy = empirical;
  } else if (policy_options.mode == 'n') {
    policy = nash;
  } else if (policy_options.mode == 'm') {
    const auto weighted_sum = [](const auto &a, const auto &b,
                                 const auto alpha) {
      auto result = a;
      std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                     [alpha](const auto &x, const auto &y) {
                       return alpha * x + (decltype(alpha))(1) - alpha * y;
                     });
      return result;
    };
    policy = weighted_sum(nash, empirical, policy_options.nash_weight);
  } else {
    throw std::runtime_error("bad policy mode.");
  }

  // TODO this is the same as using array of 9
  std::vector<double> p(policy.begin(), policy.end());
  double sum = 0;
  if (policy_options.temp != 1.0) {
    for (auto &val : p) {
      val = std::pow(val, policy_options.temp);
      sum += val;
    }
  } else {
    for (auto &val : p) {
      sum += val;
    }
  }
  if (policy_options.min_prob > 0) {
    const double l = policy_options.min_prob * sum;
    sum = 0;
    for (auto &val : p) {
      if (val < l)
        val = 0;
      sum += val;
    }
  }
  for (auto &val : p) {
    val /= sum;
  }

  const auto index = device.sample_pdf(p);

  assert(p[index] > 0);

  return index;
}

} // namespace RuntimePolicy