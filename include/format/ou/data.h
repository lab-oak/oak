#pragma once

#include <format/cart/data.h>

/*
Banned pokemon should have their movesets nullified for compatilibity with the
'flattener'
*/

namespace Learnset {

static consteval Data apply_ou_bans() {
  using enum PKMN::Data::Move;
  auto data = Learnset::Cart::data;
  for (auto &moves : data) {
    moves[static_cast<uint8_t>(Dig)] = false;
    moves[static_cast<uint8_t>(Fly)] = false;
    moves[static_cast<uint8_t>(Fissure)] = false;
    moves[static_cast<uint8_t>(HornDrill)] = false;
    moves[static_cast<uint8_t>(Minimize)] = false;
    moves[static_cast<uint8_t>(DoubleTeam)] = false;
  }
  data[150] = {};
  data[151] = {};
  return data;
}

struct OU {
  static constexpr Data data{apply_ou_bans()};
};

} // namespace Learnset

namespace Format {

using OU = FormatImpl<Learnset::OU>;

} // namespace Format
