#pragma once

#include <libpkmn/pkmn.h>

#include <vector>

namespace Train {

struct BuildTrajectory {

  struct Update {
    PKMN::Data::Species species;
    PKMN::Data::Move move;
    float prob;
  };

  std::vector<PKMN::Set> initial;
  std::vector<Update> updates;
  std::vector<PKMN::Set> terminal;
  std::vector<PKMN::Set> opponent;

  float score;
  float value;
};

}; // namespace Train
