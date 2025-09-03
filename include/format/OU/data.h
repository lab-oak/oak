#pragma once

#include <format/Cart/data.h>

/*
Banned pokemon should have their movesets nullified for compatilibity with the
'flattener'
*/

namespace Learnset {

static consteval Data apply_ou_bans() {
  using enum PKMN::Data::Move;
  auto LEARNSETS_OU_LEGAL = Learnset::Cart::data;
  for (auto &moves : LEARNSETS_OU_LEGAL) {
    moves[static_cast<uint8_t>(Dig)] = false;
    moves[static_cast<uint8_t>(Fly)] = false;
    moves[static_cast<uint8_t>(Fissure)] = false;
    moves[static_cast<uint8_t>(HornDrill)] = false;
    moves[static_cast<uint8_t>(Minimize)] = false;
    moves[static_cast<uint8_t>(DoubleTeam)] = false;
  }
  LEARNSETS_OU_LEGAL[150] = {};
  LEARNSETS_OU_LEGAL[151] = {};
  return LEARNSETS_OU_LEGAL;
}

struct OU {
  static constexpr Data data{apply_ou_bans()};
};
} // namespace Learnset

namespace Format {

using OU = FormatImpl<Learnset::OU>;

} // namespace Format
