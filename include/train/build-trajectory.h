#pragma once

#include <battle/init.h>

namespace Train {

struct ActionPolicy {
  uint16_t action;
  uint16_t policy;

  ActionPolicy() = default;
  ActionPolicy(uint16_t action, float policy)
      : action{action}, policy{static_cast<uint16_t>(
                            std::numeric_limits<uint16_t>::max() * policy)} {}
};

struct BuildTrajectory {
  std::array<ActionPolicy, 31> frames;
};

}; // namespace Train
