#pragma once

#include <format/cart-legal.h>

/*
Banned pokemon should have their movesets nullified for compatilibity with the
'flattener'
*/

namespace Format {

namespace OU {

consteval auto apply_bans() {
  using enum PKMN::Data::Move;
  auto LEARNSETS_OU_LEGAL = Cart::LEARNSETS;
  for (auto &move_pool : LEARNSETS_OU_LEGAL) {
    move_pool[static_cast<uint8_t>(Dig)] = false;
    move_pool[static_cast<uint8_t>(Fly)] = false;
    move_pool[static_cast<uint8_t>(Fissure)] = false;
    move_pool[static_cast<uint8_t>(HornDrill)] = false;
    move_pool[static_cast<uint8_t>(Minimize)] = false;
    move_pool[static_cast<uint8_t>(DoubleTeam)] = false;
  }
  for (auto &flag : LEARNSETS[150]) {
    flag = false;
  }
  for (auto &flag : LEARNSETS[151]) {
    flag = false;
  }
  return LEARNSETS_OU_LEGAL;
}

constexpr auto LEARNSETS = apply_bans();

} // namespace OU

} // namespace Format
