#pragma once

namespace Train {

#pragma pack(push, 1)
struct ActionPolicy {
  uint16_t action;
  uint16_t policy;

  ActionPolicy() = default;
  ActionPolicy(uint16_t action, float policy)
      : action{action}, policy{static_cast<uint16_t>(
                            std::numeric_limits<uint16_t>::max() * policy)} {}
};
#pragma pack(pop)

#pragma pack(push, 1)
struct BuildTrajectory {
  std::array<ActionPolicy, 31> frames;
  uint16_t eval;
  uint16_t score;
};
#pragma pack(pop)

static_assert(sizeof(BuildTrajectory) == (32 * 4));

}; // namespace Train
