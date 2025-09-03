#pragma once

#include <libpkmn/pkmn.h>

#include <vector>

/*

Data format to represent any policy method over any sequential team-building
process

We currently only use this format when rolling out the build network to
create/complete a team

Our networks assume some constraints on the team building process to make it
feasible (namely only choosing one species/move at time to make the policy space
as small as possible)

Once generated, the trajectory is saved onto disk in an encoded format which
*can* be converted back to this

But its only real purpose is to create pytorch tensors for training, so that raw
binary gets converted directly into the encoded trjaectory format.

*/

namespace Train {

namespace Build {

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
  if (action.lead) {
    std::swap(team[0], team[action.lead]);
    return;
  }
  auto &set = team[action.slot];
  if (action.move_slot) {
    set.moves[action.move_slot - 1] = action.move;
    return;
  } else {
    set.species = action.species;
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

  std::optional<std::vector<PKMN::Set>> opponent;
  float value;
  float score;
};

} // namespace Build

}; // namespace Train
