#pragma once

#include <format/OU/legal-moves.h>
#include <format/move-pool.h>
#include <train/build/trajectory.h>

namespace Encode {

using Train::Build::Action;
using Train::Build::BasicAction;

namespace Build {

template <Format::LearnsetTable LEARNSETS> struct Formatter {

  using MovePool = Format::MovePool<LEARNSETS>;

  static consteval auto get_species_move_list_size() {
    auto size = 0;
    for (const auto species : PKMN::Data::all_species) {
      const auto &move_pool{MovePool::get(species)};
      size += MovePool::size(species) > 0;
      size += MovePool::size(species);
    }

    return size;
  }

  static constexpr auto get_legal_species() {
    constexpr auto n_species = std::count_if(
        PKMN::Data::all_species.begin(), PKMN::Data::all_species.end(),
        [](const auto species) { return MovePool::size(species) > 0; });
    std::array<Species, n_species> legal_species{};
    auto it = legal_species.begin();
    for (const auto species : legal_species) {
      if (MovePool::size(species) > 0) {
        *it++ = species;
      }
    }
    return legal_species;
  }

  static constexpr auto species_move_list_size{get_species_move_list_size()};
  static constexpr auto legal_species{get_legal_species()};

  static consteval auto get_species_move_data() {
    std::array<std::array<int, PKMN::Data::all_moves.size()>,
               PKMN::Data::all_species.size()>
        table{};
    std::array<std::pair<uint8_t, uint8_t>, species_move_list_size> list{};

    std::fill(table.begin(), table.end(), -1);
    uint16_t index = 0;
    const auto go = [&index](auto species, auto move) {
      const auto s = static_cast<uint8_t>(species);
      const auto m = static_cast<uint8_t>(move);
      table[s][m] = index;
      list[index] = {s, m};
      ++index;
    };
    for (const auto species : legal_species) {
      go(s, 0);
      for (auto i = 0; i < MovePool::size(species); ++i) {
        go(s, MovePool::get(species)[i]);
      }
    }

    return std::pair<decltype(table), decltype(list)>{table, list};
  }

  // max number of actions when rolling out the build network
  static consteval int get_max_actions(const auto team_size = 6) {
    static_assert(legal_species.size() >= MovePool::max_size,
                  "This method of tightly bounding the max number of legal "
                  "team building actions likely doesn't work");
    int n = 0;
    auto sizes = MovePool::sizes;
    std::sort(sizes.begin(), sizes.end(), std::greater<uint8_t>());
    for (auto i = 0; i < team_size - 1; ++i) {
      n += (int)sizes[i];
    }
    n += legal_species.size() - (team_size - 1);
    return n;
  }

public:
  static constexpr auto n_dim{species_move_list_size};
  static constexpr auto legal_species{get_legal_species()};
  static constexpr int max_actions{get_max_actions()};
  static constexpr auto SPECIES_MOVE_DATA{get_species_move_data()};
  static constexpr auto SPECIES_MOVE_TABLE{SPECIES_MOVE_DATA.first};
  static constexpr auto SPECIES_MOVE_LIST{SPECIES_MOVE_DATA.second};

  static constexpr auto species_move_table(const auto species,
                                           const auto move) {
    return SPECIES_MOVE_TABLE[static_cast<uint8_t>(species)]
                             [static_cast<uint8_t>(move)];
  }

  static constexpr auto species_move_list(const auto index) {
    return SPECIES_MOVE_LIST[index];
  }

  static int action_index(const Action &action) {
    if (action.size() != 1) {
      throw std::runtime_error{"Compound actions not supported."};
      return -1;
    }
    const auto &a = action[0];
    const auto species = a.species;
    const auto move = a.move;
    if (a.lead > 0) {
      auto index = species_move_table(species, 0);
      assert(index >= 0);
      return index;
    }
    auto index = species_move_table(species, move);
    assert(index >= 0);
    return index;
  }

  static std::array<float, n_dim> write(const auto &team) {
    std::array<float, n_dim> input{};
    const auto go = [&input](auto species, auto move) {
      const auto index = species_move_table(species, move);
      assert(index >= 0);
      input[index] = 1.0;
    };
    for (const auto &set : team) {
      go(set.species, 0);
      for (const auto move : set.moves) {
        go(set.species, move);
      }
    }
    return input;
  }

  static std::vector<Action> get_singleton_additions(const auto &team) {
    using PKMN::Data::Move;
    using PKMN::Data::Species;
    std::vector<Action> actions;
    actions.reserve(max_actions);

    bool need_species =
        std::any_of(team.begin(), team.end(), [](const auto &set) {
          return set.species == Species::None;
        });
    auto ls = legal_species;
    auto ls_end = ls.end();
    if (need_species) {
      for (const auto &set : team) {
        ls_end = std::remove(ls.begin(), ls_end, set.species);
      }
      for (auto it = ls.begin(); it != ls_end; ++it) {
        actions.emplace_back(Action{})
      }
    }

    for (auto i = 0; i < team.size(); ++i) {
      const auto &set = team[i];
      if (set.species != Species::None) {
        species_end =
            std::remove(legal_species.begin(), species_end, set.species);

        auto empty = std::find(set.moves.begin(), set.moves.end(), Move::None);

        if (empty != set.moves.end()) {
          auto move_pool = MovePool::get(set.species);
          const auto start = move_pool.begin();
          auto end = start + MovePool::size(set.species);
          for (auto j = 0; j < set.moves.size(); ++j) {
            const auto move = set.moves[j];
            if (move == Move::None && start != end) {
              end = std::remove(start, end, move.id);
            }
          }
          actions.emplace_back(Action{BasicAction{}});
        }
      }
    }

    return actions;
  }

  static std::vector<Action> get_lead_actions()
};

using OUFormatter = Formatter<Format::OU::LEARNSETS>;

struct CompressedTrajectory {

  enum class Version : std::underlying_type_t<std::byte> {
    Default = 0,
    WithTeam = 1,
  };

  struct Header {
    Version format;
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
        updates[i++] =
            Update{OUFormatter::species_move_table(set.species, 0), 0};
        for (const auto m : set.moves) {
          if (m != Move::None) {
            updates[i++] =
                Update{OUFormatter::species_move_table(set.species, m), 0};
          }
        }
      }
    }
    assert(i + trajectory.updates.size() <= 31);

    std::transform(trajectory.updates.begin(), trajectory.updates.end() - 1,
                   updates.begin() + i, [](const auto &update) {
                     return Train::Build::Trajectory::Update{};
                   });
  }

  void write(char *data) const {}

  void read(char *data) {
    std::memcpy(reinterpret_cast<char *>(&header), data, sizeof(Header));
    data += sizeof(Header);
    std::memcpy(reinterpret_cast<char *>(updates.data()), data,
                31 * sizeof(Update));
    data += sizeof(31 * sizeof(Update));
    if (header.format == Version::WithTeam) {
      std::memcpy(reinterpret_cast<char *>(opp.data()), data,
                  sizeof(PKMN::Team));
    }
  }
};

struct TrajectoryInput {
  int64_t *action;
  int64_t *mask;
  float *policy;
  float *eval;
  float *score;

  void write(const CompressedTrajectory &traj) {

    struct Helper {
      Species species;
      std::remove_cvref_t<decltype(OUFormatter::MovePool::get(0).end())> end;
    };
  }
};

}; // namespace Build

} // namespace Encode