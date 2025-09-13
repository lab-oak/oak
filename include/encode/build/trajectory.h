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

struct TrajectoryInput {
  int64_t *action;
  int64_t *mask;
  float *policy;
  float *value;
  float *score;
  int64_t *start;
  int64_t *end;

  template <typename F = Format::OU> struct Slot {
    int species;
    std::array<Move, F::max_move_pool_size> move_pool;
    int n_moves;
    size_t move_pool_size;

    Slot() {};

    Slot(const uint8_t species)
        : species{species}, n_moves{}, max_moves{F::move_pool_size(species)},
          move_pool{F::move_pool(species)}, move_pool_size(max_moves) {}

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

  template <typename F = Format::OU> struct WriteCache {
    WriteCache() : slots{}, size{} {
      available_species = {F::legal_species.begin(), F::legal_species.end()};
    }

    std::array<Slot, 6> slots;
    int size;
    std::vector<Species> available_species;

    bool add_species(const auto s) {
      const auto found =
          std::find_if(slots.begin(), slots.end(),
                       [s](const auto &slot) { return slots.species == s; });
      if (found != slots.end()) {
        return false;
      }
      const auto empty =
          std::find_if(slots.begin(), slots.end(),
                       [](const auto &slot) { return slot.species == 0; });
      assert(empty != slots.end());
      slots[size++] = Slot{s};
      std::erase(available_species, static_cast<Species>(s));
      return true;
    }

    void add_move(const auto s, const auto m) {
      auto selected =
          std::find_if(slots.begin(), slots.end(),
                       [](const auto &slot) { return slot.species == s; });
      assert(selected != slots.end());
      *selected.add_move(m);
    }

    void write_moves() const {}

    void write_species() const {}
  };

  template <typename F = Format::OU>
  void write(const CompressedTrajectory<F> &traj) {
    constexpr float den = std::numeric_limits<uint16_t>::max();

    WriteCache<F> cache{};

    auto start = -1;
    auto last_species = 0;
    auto swap = 0;
    auto i = 0;
    for (; i < 31; ++i) {
      const auto &u = traj.updates[i];
      if ((start >= 0) && (u.p == 0)) {
        ++i;
        break;
      }
      if ((u.p > 0) && (start < 0)) {
        start = i;
      }
      const auto [s, m] = Tensorizer::species_move_list(u.action);
      if (m == 0) {
        if (!cache.add_species(s)) {
          swap = i;
        } else {
          last_species = i;
        }
      }
    }
    auto end = i;

    const auto write_invalid = [this]() {
      *action++ = -1;
      std::copy_n(mask, mask += Tensorizer<F>::max_actions, -1);
      *policy++ = 0;
    };

    const auto write_valid_species = [this, &cache](const auto &u) {};
    const auto write_valid_moves = [this, &cache](const auto &u) {};
    const auto write_swap = []() {};

    i = 0;
    for (; i < start; ++i) {
      write_invalid();
    }
    for (; i < last_species; ++i) {
      write_valid_species();
      write_valid_moves();
    }
    for (; i < swap; ++i) {
      write_moves();
    }
    for (; i < end; ++i) {
      write_swap();
    }
    for (; i < 31; ++i) {
      write_invalid();
    }
    return;
  }
};

} // namespace Build

} // namespace Encode