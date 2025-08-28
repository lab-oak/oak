#pragma once

namespace Train {

#pragma pack(push, 1)
struct EncodedBuildTrajectory2 {
  struct Update {
    // index of unrolled species/move list
    uint16_t action;
    // quantized prob of selecting the above action
    uint16_t policy;

    Update() = default;
    Update(uint16_t action, float policy)
        : action{action}, policy{static_cast<uint16_t>(
                              std::numeric_limits<uint16_t>::max() * policy)} {
      this->policy += (policy > 0) && (this->policy == 0);
    }
  };

  std::array<Update, 31> frames;
  uint16_t eval;
  uint16_t score;

  BuildTrajectory uncompress() const { BuildTrajectory traj; }
};
#pragma pack(pop)

static_assert(sizeof(EncodedBuildTrajectory2) == (32 * 4));

} // namespace Train