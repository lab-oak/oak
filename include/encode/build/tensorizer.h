#pragma once

#include <encode/build/actions.h>
#include <format/ou/data.h>
#include <format/util.h>
#include <train/build/trajectory.h>

/*

Here is where we define a tensor encoding and associated storage for build
trajectories in the game we defined in actions.h

The actions (singleton additions) are encoded as u16s

*/

namespace Encode {

using Train::Build::Action;
using Train::Build::BasicAction;

namespace Build {

template <typename F = Format::OU> struct Tensorizer {

  static consteval auto get_species_move_list_size() {
    auto size = 0;
    for (const auto species : PKMN::Data::all_species) {
      const auto &move_pool{F::move_pool(species)};
      size += F::move_pool_size(species) > 0;
      size += F::move_pool_size(species);
    }

    return size;
  }

  static constexpr auto species_move_list_size{get_species_move_list_size()};

  static consteval auto get_species_move_data() {
    std::array<std::array<int, PKMN::Data::all_moves.size()>,
               PKMN::Data::all_species.size()>
        table{};
    std::array<std::pair<uint8_t, uint8_t>, species_move_list_size> list{};

    std::array<int, PKMN::Data::all_moves.size()> invalid{};
    std::fill(invalid.begin(), invalid.end(), -1);
    std::fill(table.begin(), table.end(), invalid);

    uint16_t index = 0;
    const auto go = [&index, &table, &list](auto species, auto move) {
      const auto s = static_cast<uint8_t>(species);
      const auto m = static_cast<uint8_t>(move);
      table[s][m] = index;
      list[index] = {s, m};
      ++index;
    };
    for (const auto species : F::legal_species) {
      go(species, 0);
      for (auto i = 0; i < F::move_pool_size(species); ++i) {
        go(species, F::move_pool(species)[i]);
      }
    }

    return std::pair<decltype(table), decltype(list)>{table, list};
  }

  // max number of actions when rolling out the build network
  template <size_t team_size = 6> static consteval int get_max_actions() {
    // static_assert(legal_species.size() >= MovePool::max_size,
    //               "This method of tightly bounding the max number of legal "
    //               "team building actions likely doesn't work");
    int n = 0;
    auto sizes = F::MOVE_POOL_SIZES;
    std::sort(sizes.begin(), sizes.end(), std::greater<uint8_t>());
    for (auto i = 0; i < team_size - 1; ++i) {
      n += (int)sizes[i];
    }
    n += F::legal_species.size() - (team_size - 1);
    return n;
  }

  static constexpr auto n_dim{species_move_list_size};
  static constexpr int max_actions{get_max_actions<>()};
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
      if (species != PKMN::Data::Species::None) {
        const auto index = species_move_table(species, move);
        assert(index >= 0);
        input[index] = 1.0;
      }
    };
    for (const auto &set : team) {
      go(set.species, 0);
      for (const auto move : set.moves) {
        go(set.species, move);
      }
    }
    return input;
  }
};

} // namespace Build

} // namespace Encode