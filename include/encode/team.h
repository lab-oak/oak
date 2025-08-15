#pragma once

#include <data/move-pools.h>
#include <libpkmn/data.h>
#include <libpkmn/data/status.h>
#include <libpkmn/pkmn.h>

#include <array>
#include <cassert>

namespace Encode {

namespace Team {

consteval auto get_species_move_list_size() {
  uint16_t size = 0;
  for (auto s = 1; s <= 149; ++s) {
    for (auto m = 0; m < 166; ++m) {
      if (Data::MOVE_POOLS[s][m] || !m) {
        ++size;
      }
    }
  }
  return size;
}

constexpr auto species_move_list_size = get_species_move_list_size();

constexpr auto in_dim = species_move_list_size;
constexpr auto out_dim = in_dim;

consteval auto get_species_move_data() {
  std::array<std::array<uint16_t, 166>, 152> table{};
  std::array<std::pair<uint8_t, uint8_t>, species_move_list_size> list{};

  uint16_t index = 0;
  for (auto s = 1; s <= 149; ++s) {
    for (auto m = 0; m < 166; ++m) {
      if (Data::MOVE_POOLS[s][m] || !m) {
        table[s][m] = index;
        list[index] = {s, m};
        ++index;
      }
    }
  }
  return std::pair<decltype(table), decltype(list)>{table, list};
}

constexpr auto SPECIES_MOVE_DATA = get_species_move_data();
constexpr auto SPECIES_MOVE_TABLE = SPECIES_MOVE_DATA.first;
constexpr auto SPECIES_MOVE_LIST = SPECIES_MOVE_DATA.second;

inline constexpr auto species_move_table(const auto species, const auto move) {
  return SPECIES_MOVE_TABLE[static_cast<uint8_t>(species)]
                           [static_cast<uint8_t>(move)];
}

inline constexpr auto species_move_list(const auto index) {
  return SPECIES_MOVE_LIST[index];
}

void write(const PKMN::Team &team, float *const t) {
  for (const auto &set : team) {
    if (static_cast<bool>(set.species)) {
      t[species_move_table(set.species, 0)] = 1;
    }
    for (const auto move : set.moves) {
      if (static_cast<bool>(move)) {
        t[species_move_table(set.species, move)] = 1;
      }
    }
  }
}

Train::BuildTrajectory initial_trajectory(const auto &team) {
  Train::BuildTrajectory traj{};
  auto i = 0;
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
  return traj;
}

[[nodiscard]] bool write_policy_mask(const auto &team, float *const t) {
  bool needs_species = false;
  bool complete = true;
  uint n_pokemon = 0;
  for (const auto &set : team) {
    if (static_cast<bool>(set.species)) {
      auto n_moves = 0;
      for (const auto move : set.moves) {
        n_moves += static_cast<bool>(move);
      }
      if (n_moves < std::min(Data::move_pool_size(set.species), (uint8_t)4)) {
        complete = false;
        for (const auto move : move_pool(set.species)) {
          if (!static_cast<bool>(move)) {
            break;
          }
          t[species_move_table(set.species, move)] = 1.0;
        }
        for (const auto move : set.moves) {
          if (static_cast<bool>(move)) {
            t[species_move_table(set.species, move)] = 0.0;
          }
        }
      }
    } else {
      needs_species = true;
      complete = false;
    }
  }
  if (needs_species) {
    for (int i = 1; i <= 149; ++i) {
      t[species_move_table(i, 0)] = 1.0;
    }
    for (const auto &set : team) {
      if (static_cast<bool>(set.species)) {
        t[species_move_table(set.species, 0)] = 0.0;
      }
    }
  }
  if (complete) {
    for (const auto &set : team) {
      t[species_move_table(set.species, 0)] = 1.0;
    }
  }

  return complete;
}

void apply_index_to_team(PKMN::Team &team, auto s, auto m) {
  if (!m) {
    for (auto &set : team) {
      if (set.species == Data::Species::None) {
        set.species = static_cast<Data::Species>(s);
        return;
      } else if (set.species == static_cast<Data::Species>(s)) {
        std::swap(set, team[0]);
        return;
      }
    }
    // assert(false, "Cant add species to set/set active");
    assert(false);
  } else {
    for (auto &set : team) {
      if (set.species == static_cast<Data::Species>(s)) {
        for (auto &move : set.moves) {
          if (move == Data::Move::None) {
            move = static_cast<Data::Move>(m);
            return;
          }
        }
        // assert(false, "Cant add move to species");
        assert(false);
      }
    }
  }
}

} // namespace Team

}; // namespace Encode