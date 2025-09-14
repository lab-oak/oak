#pragma once

#include <encode/build/actions.h>
#include <encode/build/tensorizer.h>
#include <format/ou/data.h>
#include <train/build/trajectory.h>

/*

Compressed trajectory is what we actually write to disk. It assumes we are
playing the 'game' defined in actions.h

*/

namespace Encode {

using Train::Build::Action;
using Train::Build::BasicAction;

namespace Build {

using PKMN::Data::Move;
using PKMN::Data::Species;

#pragma pack(push, 1)
template <typename F = Format::OU> struct CompressedTrajectory {

  enum class Format : std::underlying_type_t<std::byte> {
    NoTeam = 0,
    WithTeam = 1,
  };
  struct Header {
    Format format{0};
    uint8_t score;
    uint16_t value;
  };
  struct Update {
    uint16_t action;
    uint16_t probability;

    Update() = default;

    Update(int a, float p) {
      assert(a >= 0);
      action = a;
      probability = (float)std::numeric_limits<uint16_t>::max() * p;
      if (p != 0 && probability == 0) {
        probability = 1;
      }
    }
  };

  Header header;
  std::array<Update, 31> updates;
  PKMN::Team opponent;

  static constexpr size_t size_no_team = sizeof(header) + sizeof(updates);
  static constexpr size_t size_with_team = size_no_team + sizeof(opponent);

  CompressedTrajectory() = default;

  CompressedTrajectory(const Train::Build::Trajectory &trajectory)
      : header{}, updates{}, opponent{} {
    using PKMN::Data::Move;
    using PKMN::Data::Species;
    auto i = 0;
    assert(trajectory.initial.size() <= 6);

    if (trajectory.opponent.has_value()) {
      header.format = Format::WithTeam;
      const auto &opp = trajectory.opponent.value();
      std::copy(opp.begin(), opp.end(), opponent.begin());
    }
    assert(trajectory.value >= 0 && trajectory.value <= 1);
    header.value = trajectory.value * std::numeric_limits<uint16_t>::max();
    if (trajectory.score.has_value()) {
      header.score = 2 * trajectory.score.value();
    } else {
      header.score = std::numeric_limits<decltype(header.score)>::max();
    }

    // encode initial team
    for (const auto &set : trajectory.initial) {
      if (set.species != Species::None) {
        updates[i++] =
            Update{Tensorizer<F>::species_move_table(set.species, 0), 0};
        for (const auto m : set.moves) {
          if (m != Move::None) {
            updates[i++] =
                Update{Tensorizer<F>::species_move_table(set.species, m), 0};
          }
        }
      }
    }
    assert(i + trajectory.updates.size() <= 31);

    // encode the updates
    assert(trajectory.updates.size() > 0);
    std::transform(trajectory.updates.begin(), trajectory.updates.end(),
                   updates.begin() + i, [](const auto &update) {
                     const auto &action = update.legal_moves[update.index];
                     assert(action.size() == 1);
                     return Update{Tensorizer<F>::species_move_table(
                                       action[0].species, action[0].move),
                                   update.probability};
                   });
  }

  void write(char *data) const {
    auto index = 0;
    std::memcpy(data + index, reinterpret_cast<const char *>(&header),
                sizeof(header));
    index += sizeof(header);
    std::memcpy(data + index, reinterpret_cast<const char *>(&updates),
                sizeof(updates));
    if (header.format == Format::WithTeam) {
      index += sizeof(updates);
      std::memcpy(data + index, reinterpret_cast<const char *>(&opponent),
                  sizeof(opponent));
    }
  }

  void read(char *data) {
    std::memcpy(reinterpret_cast<char *>(&header), data, sizeof(Header));
    data += sizeof(Header);
    std::memcpy(reinterpret_cast<char *>(&updates), data, sizeof(updates));
    if (header.format == Format::WithTeam) {
      data += sizeof(updates);
      std::memcpy(reinterpret_cast<char *>(&opponent), data, sizeof(opponent));
    }
  }
};
#pragma pack(pop)

// These two structs store what is *missing* so they can quickly write the
// actions masks
template <typename F = Format::OU> struct SetHelper {
  uint species;
  std::array<Move, F::max_move_pool_size> move_pool;
  uint n_moves;
  uint move_pool_size;

  SetHelper() {};
  SetHelper(const uint8_t species)
      : species{species}, n_moves{}, move_pool_size{F::move_pool_size(species)},
        move_pool{F::move_pool(species)} {}

  bool add_move(const auto m) {
    const auto move_pool_end =
        std::remove(move_pool.begin(), move_pool.begin() + move_pool_size,
                    static_cast<Move>(m));
    if (move_pool_end != (move_pool.begin() + move_pool_size)) {
      --move_pool_size;
      ++n_moves;
    }
  }

  bool is_complete() const { return (n_moves >= 4) || (move_pool_size == 0); }
};

template <typename F = Format::OU> struct TeamHelper {
  TeamHelper() : slots{}, size{} {
    available_species = {F::legal_species.begin(), F::legal_species.end()};
  }

  std::array<SetHelper<F>, 6> slots;
  int size;
  std::vector<Species> available_species;

  bool add_species(const auto s) {
    const auto found =
        std::find_if(slots.begin(), slots.end(),
                     [s](const auto &slot) { return slot.species == s; });
    if (found != slots.end()) {
      return false;
    }
    const auto empty =
        std::find_if(slots.begin(), slots.end(),
                     [](const auto &slot) { return slot.species == 0; });
    assert(empty != slots.end());
    slots[size++] = SetHelper<F>{s};
    // std::erase(available_species, static_cast<Species>(s));
    return true;
  }

  void add_move(const auto s, const auto m) {
    auto selected =
        std::find_if(slots.begin(), slots.end(),
                     [s](const auto &slot) { return slot.species == s; });
    assert(selected != slots.end());
    (*selected).add_move(m);
  }

  auto prefill_and_get_bounds(const auto &updates) {

    std::tuple<uint, uint, uint, uint> data{};
    bool started = false;
    auto &[start, full, swap, end] = data;
    auto i = 0;

    for (; i < 31; ++i) {
      const auto &u = updates[i];
      if ((start >= 0) && (u.probability == 0)) {
        break;
      }
      if ((u.probability > 0) && !started) {
        started = true;
        start = i;
      }
      const auto [s, m] = Tensorizer<F>::species_move_list(u.action);
      if (m == 0) {
        if (!add_species(s)) {
          swap = i + 1;
        } else {
          full = i + 1;
        }
      }
    }
    end = i + 1;
    size = 0;

    return data;
  }

  void apply(const auto &update) {
    const auto [s, m] = Tensorizer<F>::species_move_list(update.action);
    if (m == 0) {
      std::erase(available_species, static_cast<Species>(s));
      ++size;
    } else {
      add_move(s, m);
    }
  }
};

struct TrajectoryInput {
  int64_t *action;
  int64_t *mask;
  float *policy;
  float *value;
  float *score;
  int64_t *start;
  int64_t *end;

  void step_update(auto i) {
    action += i;
    mask += i * Tensorizer<F>::max_actions;
    policy += i;
  }

  template <typename F = Format::OU>
  void write(const CompressedTrajectory<F> &traj) {

    constexpr float den = std::numeric_limits<uint16_t>::max();
    using Tensorizer<F>::max_actions;

    TeamHelper<F> helper{};

    const auto [start, full, swap, end] = helper.prefill(traj.updates);

    const auto write_valid = [this, &helper](const auto &u) -> int {
      *action++ = u.action;
      *policy++ = u.probability / den;
      old_mask = mask;
      for (auto i = 0; i < helper.size; ++i) {
        const auto &set = helper.slots[i];
        std::transform(
            set.move_pool.begin(), set.move_pool.begin() + set.move_pool_size,
            mask, [&set](const auto m) {
              return Tensorizer<F>::species_move_table(set.species, m);
            });
        mask += set.move_pool_size;
      }
      std::transform(helper.available_species.begin(),
                     helper.available_species.end(), mask,
                     [](const auto s) { return species_move_table(s, 0); });
      mask = old_mask + max_actions;
      helper.apply(u);
    };

    const auto write_valid_full = [this, &helper](const auto &u) {
      *action++ = u.action;
      *policy++ = u.probability / den;
      old_mask = mask;
      for (auto i = 0; i < helper.size; ++i) {
        const auto &set = helper.slots[i];
        std::transform(
            set.move_pool.begin(), set.move_pool.begin() + set.move_pool_size,
            mask, [&set](const auto m) {
              return Tensorizer<F>::species_move_table(set.species, m);
            });
        mask += set.move_pool_size;
      }
      mask = old_mask + max_actions;
      helper.apply(u);
    };

    std::fill(mask, mask + (31 * max_actions), -1);
    std::fill(action, action + 31, -1);
    std::fill(policy, policy + 31, 0.0f);
    step_update(start);

    i = start;
    for (; i < full; ++i) {
      const auto &u = traj.updates[i];
      old_mask = mask;
      write_valid(u);
      write_valid_species();
      mask = old_mask + max_actions;
    }
    for (; i < swap; ++i) {
      const auto &u = traj.updates[i];
      old_mask = mask;
      write_valid(u);
      mask = old_mask + max_actions;
    }
    for (; i < end; ++i) {
      write_swap();
    }

    *(this->start)++ = start;
    *(this->end)++ = end;
    *value++ = traj.value / den;
    if (traj.score == std::numeric_limits<uint16_t>::max()) {
      *score++ = -1;
    } else {
      *score++ = traj.score / 2.0;
    }

    return;
  }
};

} // namespace Build

} // namespace Encode