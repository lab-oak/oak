#pragma once

namespace Train {

#pragma pack(push, 1)
struct ActionPolicy {
  // index of unrolled species/move list
  uint16_t action;
  // quantized prob of selecting the above action
  uint16_t policy;

  ActionPolicy() = default;
  ActionPolicy(uint16_t action, float policy)
      : action{action}, policy{static_cast<uint16_t>(
                            std::numeric_limits<uint16_t>::max() * policy)} {
    this->policy += (policy > 0) && (this->policy == 0);
  }
};

struct BuildTrajectory {
  std::array<ActionPolicy, 31> frames;
  uint16_t eval;
  uint8_t score;
  uint8_t size;
};
#pragma pack(pop)

static_assert(sizeof(BuildTrajectory) == (32 * 4));

}; // namespace Train
