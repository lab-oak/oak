#pragma once

#include <libpkmn/pkmn.h>

#include <vector>

namespace Train {

namespace TeamBuilding {

using PKMN::Data::Move;
using PKMN::Data::Species;

struct BasicAction {
  uint lead;
  uint slot;
  uint move_slot;
  Species species;
  Move move;
};

using Action = std::vector<BasicAction>;

// this can turn any team into any other team. It is up to the team building
// formulation to define what diffs are allowed.
void apply_basic_action(auto &team, const BasicAction &action) {
  if (lead) {
    std::swap(team[0], team[lead]);
    return;
  }
  auto &set = team[slot];
  if (move_slot) {
    set[move_slot - 1] = move;
    return;
  } else {
    set.species = species;
    return;
  }
}

void apply_action(auto &team, const Action &action) {
  for (const auto &basic : action) {
    apply_basic_action(team, basic);
  }
}

struct Trajectory {

  struct Update {
    std::vector<Action> legal_moves;
    uint index;
    float probability;
  };

  std::vector<PKMN::Set> initial;
  std::vector<Update> updates;
  std::vector<PKMN::Set> terminal;

  std::vector<PKMN::Set> opponent;
  float value;
  float score;
};

} // namespace TeamBuilding

}; // namespace Train
