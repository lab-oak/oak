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

  template <typename F = Format::OU>
  void write(const CompressedTrajectory<F> &traj) {
    constexpr float den = std::numeric_limits<uint16_t>::max();

    struct MaskCache {
      int species;
      int max_moves;
      std::remove_cvref_t<decltype(F::move_pool(0))> move_pool;
      int n_moves;
      size_t move_pool_size;

      MaskCache() = default;

      MaskCache(const uint8_t species)
          : species{species}, n_moves{}, max_moves{F::move_pool_size(species)},
            move_pool{F::move_pool(species)}, move_pool_size(max_moves) {}
    };

    std::array<MaskCache, 6> caches{};
    auto n_cache = 0;
    std::vector<PKMN::Data::Species> available_species(F::legal_species.begin(),
                                                       F::legal_species.end());
    bool can_add_species = true;
    bool done = false;
    bool started = false;

    for (auto i = 0; i < 31; ++i) {
      const auto &update = traj.updates[i];

      int64_t _action = -1;
      std::array<int64_t, Tensorizer<F>::max_actions> _mask;
      std::fill(_mask.begin(), _mask.end(), -1);
      float _policy = 0;

      if (update.probability > 0 && !started) {
        started = true;
        *start++ = i;
      }
      if (started && update.probability == 0 && !done) {
        done = true;
        *end++ = i;
      }

      if (!done) {
        _action = update.action;
        if (started) {
          // set info for writing
          _policy = update.probability / den;
          auto mask_index = 0;
          for (const auto &cache : caches) {
            if ((cache.species != 0) &&
                (cache.n_moves < std::min(4, cache.max_moves))) {
              std::transform(
                  cache.move_pool.begin(),
                  cache.move_pool.begin() + cache.move_pool_size,
                  _mask.begin() + mask_index, [&cache](const auto move) {
                    const auto action =
                        Tensorizer<F>::species_move_table(cache.species, move);
                    assert(action >= 0);
                    return action;
                  });
              mask_index += cache.move_pool_size;
            } else {
            }
          }
          if (can_add_species) {
            // copy avaiable mons to _mask
            std::transform(available_species.begin(), available_species.end(),
                           _mask.begin() + mask_index, [](const auto species) {
                             return Tensorizer<F>::species_move_table(species,
                                                                      0);
                           });
            mask_index += available_species.size();
          }
           // make the selected action the first one in the mask
          std::swap(*_mask.begin(), *std::find(_mask.begin(), _mask.end(), update.action));
        }

        // update stats
        const auto [s, m] = Tensorizer<F>::species_move_list(update.action);
        auto it =
            std::find_if(caches.begin(), caches.begin() + n_cache,
                         [s](const auto &cache) { return cache.species == s; });
        if (m == 0) {
          // add species
          if (it == caches.begin() + n_cache) {
            caches[n_cache++] = MaskCache{s};
            std::erase(available_species, static_cast<PKMN::Data::Species>(s));
          } else {
            can_add_species = false;
          }
        } else {
          // add move
          assert(it != (caches.begin() + n_cache));
          auto &cache = *it;
          auto &move_pool = cache.move_pool;
          auto x = std::find(move_pool.begin(),
                             move_pool.begin() + cache.move_pool_size,
                             static_cast<PKMN::Data::Move>(m));
          if (x != (move_pool.begin() + cache.move_pool_size)) {
            std::swap(move_pool[cache.move_pool_size - 1], *x);
            --cache.move_pool_size;
            ++cache.n_moves;
          } else {
            // we should not be able to add a move that is not present
            assert(false);
          }
        }
      }

      // write info
      *action++ = _action;
      std::copy(_mask.begin(), _mask.end(), mask);
      mask += Tensorizer<F>::max_actions;
      *policy++ = _policy;
    }

    if (!done) {
      *end++ = 31;
    }

    *value++ = traj.header.value / den;
    if (traj.header.score ==
        std::numeric_limits<decltype(traj.header.score)>::max()) {
      *score++ = -1;
    } else {
      *score++ = traj.header.score / 2.0;
    }
  }
};

} // namespace Build

} // namespace Encode