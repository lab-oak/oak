#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/strings.h>

#include <cassert>

/*

Actions in battle are encoded as either a move or a species. All moves are
encoded except for None and Struggle. The latter is because Struggle is only
possible when theres one move, so no policy inference is needed.

The rest is self explanatory. Moves are encoded as themselves, and switches as
the incoming Pokemon's species.

This encoding keeps the number of logits small (n_dim) and only 9 actions max
are legal, so we don't have to compute the entire logit layer.

*/

namespace Encode::Battle::Policy {

// all moves besides struggle, all pokemon. No None.
constexpr auto n_dim = 164 + 151;

uint16_t get_index(const PKMN::Side &side, auto choice) {
  const auto choice_type = choice & 3;
  const auto choice_data = choice >> 2;
  switch (choice_type) {
  case 1: {
    assert(choice_data >= 0 && choice_data <= 4);
    auto moveid =
        static_cast<uint16_t>(side.stored().moves[choice_data - 1].id);
    // assert(moveid < static_cast<uint16_t>(Data::Move::Struggle));
    if (moveid == 0) {
      return 0;
    }
    return moveid - 1;
  }
  case 2: {
    assert(choice_data >= 2 && choice_data <= 6);
    const auto &pokemon = side.get(choice_data);
    auto species = static_cast<uint16_t>(pokemon.species);
    assert(species <= 151);
    return 164 + species - 1;
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

consteval auto get_dim_labels() {
  std::array<std::array<char, 13>, n_dim> result{};
  const auto copy = [](const auto &src, auto &dest) {
    for (auto i = 0; i < src.size(); ++i) {
      dest[i] = src[i];
    }
  };
  auto index = 0;
  for (auto i = 0; i < 164; ++i) {
    copy(PKMN::Data::MOVE_CHAR_ARRAY[i + 1], result[index + i]);
  }
  index += 164;
  for (auto i = 0; i < 151; ++i) {
    copy(PKMN::Data::SPECIES_CHAR_ARRAY[i + 1], result[index + i]);
  }
  return result;
}

constexpr auto dim_labels = get_dim_labels();

} // namespace Encode::Battle::Policy