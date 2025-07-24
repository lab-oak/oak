#pragma once

#include <array>
#include <cstdint>

namespace Train {

struct Target {
  std::array<float, 9> p1_empirical;
  std::array<float, 9> p1_nash;
  std::array<float, 9> p2_empirical;
  std::array<float, 9> p2_nash;
  float empirical_value;
  float nash_value;
  float score;
};

} // namespace Train