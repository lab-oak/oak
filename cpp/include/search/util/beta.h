#pragma once

template <class RNG> double gamma_sample(double a, RNG &rng) {
  static std::normal_distribution<double> norm(0.0, 1.0);
  static std::uniform_real_distribution<double> uni(0.0, 1.0);

  if (a < 1.0) {
    // boost trick
    double u = uni(rng);
    return gamma_sample(a + 1.0, rng) * std::pow(u, 1.0 / a);
  }

  double d = a - 1.0 / 3.0;
  double c = 1.0 / std::sqrt(9.0 * d);

  while (true) {
    double x = norm(rng);
    double v = 1.0 + c * x;
    if (v <= 0)
      continue;
    v = v * v * v;

    double u = uni(rng);

    if (u < 1 - 0.0331 * x * x * x * x)
      return d * v;

    if (std::log(u) < 0.5 * x * x + d * (1 - v + std::log(v)))
      return d * v;
  }
}

template <class RNG> double beta_sample(double alpha, size_t N, RNG &rng) {
  double beta = double(N) - alpha;
  double x = gamma_sample<RNG>(alpha, rng);
  double y = gamma_sample<RNG>(beta, rng);
  return x / (x + y);
}

template <class RNG> double fast_beta(double alpha, size_t N, RNG &rng) {
  double beta = N - alpha;
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  // Johnk's method
  while (true) {
    double u = uni(rng);
    double v = uni(rng);

    double x = std::pow(u, 1.0 / alpha);
    double y = std::pow(v, 1.0 / beta);

    if (x + y <= 1.0)
      return x / (x + y);
  }
}