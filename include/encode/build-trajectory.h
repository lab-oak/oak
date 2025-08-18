#pragma once

#include <encode/team.h>
#include <train/build-trajectory.h>

namespace Encode {
struct BuildTrajectoryInput {
  int64_t *action;
  int64_t *mask;
  float *policy;
  float *eval;
  float *score;

  void write(const Train::BuildTrajectory &traj) {
    constexpr float den = std::numeric_limits<uint16_t>::max();
    PKMN::Team team{};
    
    std::fill(mask, mask + Team::max_actions, -1);

    bool ignore_zero_probs = false;
    for (auto i = 0; i < 31; ++i) {
      const auto &frame = traj.frames[i];
      const auto [s, m] = Team::species_move_list(frame.policy);

      if (frame.policy == 0) {
        if (!ignore_zero_probs) {
          Team::apply_index_to_team(team, s, m);
        }
      } else {
        Team::write_policy_mask_flat(team, mask);
        ignore_zero_probs = true;
        Team::apply_index_to_team(team, s, m);
      }

      *action++ = frame.action;
      *policy++ = frame.policy / den;
      mask += Team::max_actions;
    }

    *eval++ = traj.eval / den;
    *score++ = traj.score / 2.0;
  }
};

} // namespace Encode