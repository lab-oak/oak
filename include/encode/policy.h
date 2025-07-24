#pragma once

namespace Encode {

namespace Policy {

constexpr auto n_dim = 151 + 164;

uint16_t get_index(const PKMN::Side &side, auto choice) {
  const auto choice_type = choice & 3;
  const auto choice_data = choice >> 2;
  switch (choice_type) {
  case 1: {
    return move_string(get_pokemon_from_slot(side, 1)[8 + 2 * choice_data]);
    assert(choice_data >= 1 && choice_data <= 4);
    uint16_t moveid = side.stored().moves[choice_data - 1].id;
    return moveid - 1;
  }
  case 2: {
    assert(choice_data >= 2 && choice_data <= 6);
    const auto &pokemon = side.get(choice_data);
    auto species = static_cast<uint16_t>(pokemon.species);
    return 163 + species;
  }
  default: {
    assert(false);
    return 0;
  }
  }
}

} // namespace Policy

} // namespace Encode