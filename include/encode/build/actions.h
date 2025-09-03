#pragma once

#include <format/OU/data.h>
#include <train/build/trajectory.h>

/*

Here we define what actions are legal in a given 'format' (i.e. Smogon standard
tiers)

Here is where we enforce species clause and use the format data to know which
pokemon are legal and what their moves are

*/

namespace Encode {

namespace Build {

template <typename F = Format::OU> struct Actions {

  static std::vector<Train::Build::Action>
  get_singleton_additions(const auto &team) {

    using PKMN::Data::Move;
    using PKMN::Data::Species;
    using Train::Build::Action;
    using Train::Build::BasicAction;

    std::vector<Action> actions{};

    // add a unique pokemon to the first empty slot
    auto empty_slot =
        std::find_if(team.begin(), team.end(), [](const auto &set) {
          return set.species == Species::None;
        });
    if (empty_slot != team.end()) {
      auto ls = F::legal_species;
      auto ls_end = ls.end();
      for (const auto &set : team) {
        ls_end = std::remove(ls.begin(), ls_end, set.species);
      }
      std::transform(ls.begin(), ls_end, std::back_inserter(actions),
                     [&empty_slot](const auto species) {
                       return Action{BasicAction{
                           0, std::distance(team.begin(), empty_slot), 0,
                           species, Move::None}};
                     });
    }

    for (const auto &set : team) {
      // add a unique move to the first empty move slot
      if (set.species != Species::None) {
        auto empty = std::find(set.moves.begin(), set.moves.end(), Move::None);
        if (empty != set.moves.end()) {
          auto move_pool = F::move_pool(set.species);
          const auto start = move_pool.begin();
          auto end = start + F::move_pool_size(set.species);
          for (const auto move : set.moves) {
            end = std::remove(start, end, move);
          }
          std::transform(start, end, std::back_inserter(actions),
                         [&team, &set, &empty](const auto move) {
                           return Action{BasicAction{
                               0, std::distance(&team[0], &set),
                               std::distance(set.moves.begin(), empty),
                               set.species, move}};
                         });
        }
      }
    }

    return actions;
  }

  static auto get_lead_actions(const auto &team) {
    using PKMN::Data::Move;
    using PKMN::Data::Species;
    using Train::Build::Action;
    using Train::Build::BasicAction;

    std::vector<Train::Build::Action> actions{};
    for (auto i = 1; i < team.size(); ++i) {
      const auto &set = team[i];
      if (set.species != Species::None) {
        actions.emplace_back(
            Action{BasicAction{i, 0, 0, set.species, Move::None}});
      }
    }
    return actions;
  }
};

} // namespace Build

} // namespace Encode