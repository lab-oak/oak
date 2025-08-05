#pragma once

namespace Train {

#pragma pack(push, 1)
struct ActionPolicy {
  // index of unrolled species/move list - see
  uint16_t action;
  // quantized prob of selecting the above action
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

struct BuildTrajectoryInput {
  int64_t *action;
  float *policy;
  float *eval;
  float *score;

  void write(const BuildTrajectory &traj) {
    constexpr float den = std::numeric_limits<uint16_t>::max();

    for (auto i = 0; i < 31; ++i) {
      *action++ = traj.frames[i].action;
      *policy++ = traj.frames[i].policy / den;
    }
    *eval++ = traj.eval / den;
    *score++ = traj.score / 2;
  }
};

static_assert(sizeof(BuildTrajectory) == (32 * 4));

}; // namespace Train
