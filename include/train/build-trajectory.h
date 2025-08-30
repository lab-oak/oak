#pragma once

#include <libpkmn/pkmn.h>

#include <vector>

namespace Train {

namespace TeamBuilding {

using PKMN::Data::Species;
using PKMN::Data::Move;

struct Action {
    Species species;
    Move move;
    uint swap = 0;
};

void apply_action(auto &team, Action action) {
  const auto [s, m] = action;
  if (action.swap) {
    std::swap(team[action.swap], team[0]);
    return;
  }
  assert(s != Species::None);
  if (m == Move::None) {
    for (auto &set : team) {
      if (set.species == PKMN::Data::Species::None) {
        set.species = s;
        return;
      }
    }
    throw std::runtime_error{
        std::format("Can't apply species {} to team", PKMN::species_string(s))};

  } else {
    for (auto &set : team) {
      if (set.species == static_cast<PKMN::Data::Species>(s)) {
        for (auto &move : set.moves) {
          if (move == Move::None) {
            move = m;
            return;
          }
        }
      }
    }
    throw std::runtime_error{
        std::format("Can't apply {} to {}, species not found",
                    PKMN::move_string(m), PKMN::species_string(s))};
  }
}

struct BuildTrajectory {

  struct Update {
    std::vector<Action> legal_moves;
    Action selected;
    float probability;
  };

  std::vector<PKMN::Set> initial;
  std::vector<Update> updates;
  std::vector<PKMN::Set> terminal;

  std::vector<PKMN::Set> opponent;
  float value;
  float score;
};

}

}; // namespace Train
