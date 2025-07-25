#pragma once

#include <libpkmn/data/moves.h>

namespace Encode {

namespace Policy {

constexpr auto n_dim = 151 + 164;

uint16_t get_index(const PKMN::Side &side, auto choice) {
  const auto choice_type = choice & 3;
  const auto choice_data = choice >> 2;
  switch (choice_type) {
  case 1: {
    assert(choice_data >= 0 && choice_data <= 4);
    auto moveid =
        static_cast<uint16_t>(side.stored().moves[choice_data - 1].id);
    // assert(moveid < static_cast<uint16_t>(Data::Move::Struggle));
    return moveid - 1;
  }
  case 2: {

    assert(choice_data >= 2 && choice_data <= 6);
    const auto &pokemon = side.get(choice_data);
    auto species = static_cast<uint16_t>(pokemon.species);
    assert(species <= 151);
    return 163 + species;
  }
  case 0: {
    return 0;
  }
  default: {
    assert(false);
    return 0;
  }
  }
}

} // namespace Policy

} // namespace Encode