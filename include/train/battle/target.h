#pragma once

#include <array>

namespace Train {

namespace Battle {

struct Target {
  uint32_t iterations;
  std::array<float, 9> p1_empirical;
  std::array<float, 9> p1_nash;
  std::array<float, 9> p2_empirical;
  std::array<float, 9> p2_nash;
  float empirical_value;
  float nash_value;
  float score;
};

} // namespace Battle

} // namespace Train