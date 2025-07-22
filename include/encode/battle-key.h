#pragma once

#include <encode/battle.h>

namespace Encode {
consteval uint8_t pp_byte(uint64_t v) noexcept {
  // Shift so byte1 -> low byte, byte3 -> bits 24..31, etc.
  uint64_t t = (v >> 8) & 0x00FF00FF00FF00FFull;

  uint8_t r = 0;
  r |= (!!((t >> 0) & 0xFFu)) << 0;  // byte1
  r |= (!!((t >> 16) & 0xFFu)) << 1; // byte3
  r |= (!!((t >> 32) & 0xFFu)) << 2; // byte5
  r |= (!!((t >> 48) & 0xFFu)) << 3; // byte7
  return r;
}

static_assert(pp_byte(std::bit_cast<uint64_t>(std::array<uint8_t, 8>{
                  1, 0, 1, 1, 1, 1, 1, 1})) == 0b00001110);

static_assert(pp_byte(std::bit_cast<uint64_t>(std::array<uint8_t, 8>{
                  1, 1, 1, 1, 1, 1, 1, 0})) == 0b00000111);

static_assert(pp_byte(std::bit_cast<uint64_t>(std::array<uint8_t, 8>{
                  0, 0, 0, 0, 0, 0, 0, 0})) == 0b00000000);


uint8_t pokemon_key(const PKMN::Pokemon& pokemon, const uint8_t sleep) {
    uint8_t byte = pp_byte(std::bit_cast<uint64_t>(pokemon.moves));
    if (pokemon.status) {
        byte |= (Encode::Status::get_status_index(pokemon.status, sleep) + 1) << 4;
    }
    return byte;
}
}