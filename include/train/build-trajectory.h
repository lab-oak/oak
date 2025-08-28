#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/pkmn.h>

#include <vector>

namespace Train {

struct BuildTrajectory {

  struct Update {
    PKMN::Data::Species species;
    PKMN::Data::Move move;
    float prob;
    std::array<int, 100> mask;
  };

  std::vector<Update> updates;

  float score;
  float value;

  std::vector<PKMN::PokemonInit> team;
  std::vector<PKMN::PokemonInit> opp_team;
};

}; // namespace Train
