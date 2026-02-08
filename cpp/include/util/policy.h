#pragma once

#include <util/strings.h>

namespace RuntimePolicy {

struct Options {
  std::string mode = "e";
  double temp = 1;
  double min = 0;
  double nash_weight = .5;
};

enum class Mode : char {
  empirical = 'e',
  nash = 'n',
  argmax = 'x',
};

std::array<double, 9> get_policy(const auto &empirical, const auto &nash,
                                 const auto &options) {
  std::array<double, 9> policy{};

  const auto mode_split = Parse::split(options.mode, '-');

  for (std::string_view word : mode_split) {
    const double w =
        (word.size() > 1) ? std::stod(std::string(word.substr(1))) : 1.0;

    switch (static_cast<Mode>(word[0])) {
    case Mode::empirical: {
      std::transform(empirical.begin(), empirical.end(), policy.begin(),
                     policy.begin(),
                     [w](double e, double p) { return p + w * e; });
      break;
    }
    case Mode::nash: {
      std::transform(nash.begin(), nash.end(), policy.begin(), policy.begin(),
                     [w](double n, double p) { return p + w * n; });
      break;
    }
    case Mode::argmax: {
      const auto it = std::max_element(empirical.begin(), empirical.end());
      const size_t idx = std::distance(empirical.begin(), it);
      policy[idx] += w;
      break;
    }
    default: {
      throw std::runtime_error{"RuntimePolicy: invalid mode char: " + word[0]};
      break;
    }
    }
  }

  if (options.temp != 1) {
    double sum = 0;
    for (auto &x : policy) {
      x = std::pow(x, options.temp);
      sum += x;
    }
    for (auto &x : policy) {
      x /= sum;
    }
  }

  double sum = 0;
  for (auto &x : policy) {
    if (x < options.min) {
      x = 0;
    }
    sum += x;
  }
  if (sum == 0) {
    throw std::runtime_error{"RuntimePolicy: zero policy"};
  }
  for (auto &x : policy) {
    x /= sum;
  }

  return policy;
}

int process_and_sample(auto &device, const auto &empirical, const auto &nash,
                       const auto &policy_options) {
  const auto p = get_policy(empirical, nash, policy_options);
  const auto index = device.sample_pdf(p);
  assert(p[index] > 0);
  return index;
}

} // namespace RuntimePolicy