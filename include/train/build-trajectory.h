#pragma once

#include <encode/team.h>

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
  uint16_t score;
};
#pragma pack(pop)

static_assert(sizeof(BuildTrajectory) == (32 * 4));

[[nodiscard]] auto initial_trajectory(const auto &team)
    -> std::pair<Train::BuildTrajectory, int> {
  Train::BuildTrajectory traj{};
  int i = 0;
  for (const auto &set : team) {
    if (set.species != Data::Species::None) {
      traj.frames[i++] =
          Train::ActionPolicy{species_move_table(set.species, 0), 0};
      for (const auto move : set.moves) {
        if (move != Data::Move::None) {
          traj.frames[i++] =
              Train::ActionPolicy{species_move_table(set.species, move), 0};
        }
      }
    }
  }
  return {traj, i};
}

}; // namespace Train
