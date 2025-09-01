#pragma once

#include <encode/team.h>
#include <train/build-trajectory.h>

namespace Encode {

namespace TeamBuilding {

struct CompressedTrajectory {

  enum class Format : std::underlying_type_t<std::byte> {
    Default = 0,
    WithTeam = 1,
  };

  struct Header {
    Format format;
    uint8_t score;
    uint16_t eval;
  };

  Header header;
  struct Update {
    uint16_t action;
    uint16_t probability;
  };
  std::array<Update, 31> updates;
  PKMN::Team opp;

  CompressedTrajectory(const Train::Build::Trajectory &trajectory) {
    using PKMN::Data::Move;
    using PKMN::Data::Species;
    auto i = 0;
    assert(trajectory.initial.size() <= 6);
    for (const auto &set : trajectory.initial) {
      if (set.species != Species::None) {
        updates[i++] = {Encode::Team::species_move_table(set.species, 0), 0};
        for (const auto m : set.moves) {
          if (m != Move::None) {
            updates[i++] = {Encode::Team::species_move_table(set.species, m),
                            0};
          }
        }
      }
    }
    assert(i + trajectory.updates.size() <= 31);

    std::transform(trajectory.updates.start(), trajectory.updates.end() - 1, 
                   updates.start() + i, [](const auto &update) {
                    Encode::Team::species_move_table
                   });
  }

  void write(char *data) const {}

  void read(char *data) {
    std::memcpy(reinterpret_cast<char *>(&header), data, sizeof(Header));
    data += sizeof(Header);
    std::memcpy(reinterpret_cast<char *>(updates.data()), data,
                31 * sizeof(Update));
    data += sizeof(31 * sizeof(Update));
    if (header.format == Format::WithTeam) {
      std::memcpy(reinterpret_cast<char *>(opp.data()), data,
                  sizeof(PKMN::Team));
    }
  }
};

struct BuildTrajectoryInput {
  int64_t *action;
  int64_t *mask;
  float *policy;
  float *eval;
  float *score;
  int64_t *size;

  void write(const Train::BuildTrajectory &traj) {
    constexpr float den = std::numeric_limits<uint16_t>::max();
    std::vector<PKMN::PokemonInit> team{};
    team.resize(traj.size);

    bool ignore_zero_probs = false;
    for (auto i = 0; i < 31; ++i) {
      const auto &frame = traj.frames[i];
      const auto [s, m] = Team::species_move_list(frame.action);
      std::fill(mask, mask + Team::max_actions, -1);

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
    *size++ = traj.size;
  }
};

} // namespace TeamBuilding

} // namespace Encode