#pragma once

#include <format/OU/data.h>

namespace Encode {

namespace Build {

template <typename F = Format::OU> struct Actions {
  static std::vector<Action> get_singleton_additions(const auto &team) {
    using PKMN::Data::Move;
    using PKMN::Data::Species;
    std::vector<Action> actions;
    actions.reserve(max_actions);

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
      for (auto it = ls.begin(); it != ls_end; ++it) {
        actions.emplace_back(Action{BasicAction{
            0, std::distance(team.begin(), empty_slot), 0, *it, Move::None}});
      }
    }

    for (auto i = 0; i < team.size(); ++i) {
      const auto &set = team[i];
      if (set.species != Species::None) {
        auto empty = std::find(set.moves.begin(), set.moves.end(), Move::None);

        if (empty != set.moves.end()) {
          auto move_pool = MovePool::get(set.species);
          const auto start = move_pool.begin();
          auto end = start + MovePool::size(set.species);
          for (auto j = 0; j < set.moves.size(); ++j) {
            const auto move = set.moves[j];
            if (move == Move::None && start != end) {
              end = std::remove(start, end, move);
            }
          }
          actions.emplace_back(
              Action{BasicAction{0, i, std::distance(set.moves.begin(), empty),
                                 set.species, Move::None}});
        }
      }
    }

    return actions;
  }

  static std::vector<Action> get_lead_actions() { return {}; }
};
} // namespace Build

} // namespace Encode