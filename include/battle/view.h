#pragma once

#include <pkmn.h>

#include <data/layout.h>
#include <data/moves.h>
#include <data/species.h>
#include <data/status.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace View {

using namespace Layout;

struct Stats {
  uint8_t bytes[10];

  uint16_t &hp() noexcept { return *reinterpret_cast<uint16_t *>(bytes + 0); }
  const uint16_t &hp() const noexcept {
    return *reinterpret_cast<const uint16_t *>(bytes + 0);
  }

  uint16_t &atk() noexcept { return *reinterpret_cast<uint16_t *>(bytes + 2); }
  const uint16_t &atk() const noexcept {
    return *reinterpret_cast<const uint16_t *>(bytes + 2);
  }

  uint16_t &def() noexcept { return *reinterpret_cast<uint16_t *>(bytes + 4); }
  const uint16_t &def() const noexcept {
    return *reinterpret_cast<const uint16_t *>(bytes + 4);
  }

  uint16_t &spe() noexcept { return *reinterpret_cast<uint16_t *>(bytes + 6); }
  const uint16_t &spe() const noexcept {
    return *reinterpret_cast<const uint16_t *>(bytes + 6);
  }

  uint16_t &spc() noexcept { return *reinterpret_cast<uint16_t *>(bytes + 8); }
  const uint16_t &spc() const noexcept {
    return *reinterpret_cast<const uint16_t *>(bytes + 8);
  }
};

struct MoveSlot {
  Data::Move id;
  uint8_t pp;
};

struct Pokemon {
  uint8_t bytes[24];

  Stats &stats() noexcept { return *reinterpret_cast<Stats *>(bytes + 0); }
  const Stats &stats() const noexcept {
    return *reinterpret_cast<const Stats *>(bytes + 0);
  }

  std::array<MoveSlot, 4> &moves() noexcept {
    return *reinterpret_cast<std::array<MoveSlot, 4> *>(bytes + 10);
  }
  const std::array<MoveSlot, 4> &moves() const noexcept {
    return *reinterpret_cast<const std::array<MoveSlot, 4> *>(bytes + 10);
  }

  MoveSlot &moves(size_t i) noexcept { return moves()[i]; }
  const MoveSlot &moves(size_t i) const noexcept { return moves()[i]; }

  uint16_t &hp() noexcept { return *reinterpret_cast<uint16_t *>(bytes + 18); }
  const uint16_t &hp() const noexcept {
    return *reinterpret_cast<const uint16_t *>(bytes + 18);
  }

  float percent() const noexcept {
    // Return ceil of current hp divided by max hp times 100
    return std::ceil(100.0f * hp() / stats().hp());
  }

  Data::Status &status() noexcept {
    return *reinterpret_cast<Data::Status *>(bytes + 20);
  }
  const Data::Status &status() const noexcept {
    return *reinterpret_cast<const Data::Status *>(bytes + 20);
  }

  Data::Species &species() noexcept {
    return *reinterpret_cast<Data::Species *>(bytes + 21);
  }
  const Data::Species &species() const noexcept {
    return *reinterpret_cast<const Data::Species *>(bytes + 21);
  }

  uint8_t &types() noexcept { return bytes[22]; }
  const uint8_t &types() const noexcept { return bytes[22]; }

  uint8_t &level() noexcept { return bytes[23]; }
  const uint8_t &level() const noexcept { return bytes[23]; }
};

// Align Volatiles for uint64_t access
struct Volatiles {
  uint64_t bits;

  bool bide() const { return bits & (1ULL << 0); }
  bool thrashing() const { return bits & (1ULL << 1); }
  bool multi_hit() const { return bits & (1ULL << 2); }
  bool flinch() const { return bits & (1ULL << 3); }
  bool charging() const { return bits & (1ULL << 4); }
  bool binding() const { return bits & (1ULL << 5); }
  bool invulnerable() const { return bits & (1ULL << 6); }
  bool confusion() const { return bits & (1ULL << 7); }
  bool mist() const { return bits & (1ULL << 8); }
  bool focus_energy() const { return bits & (1ULL << 9); }
  bool substitute() const { return bits & (1ULL << 10); }
  bool recharging() const { return bits & (1ULL << 11); }
  bool rage() const { return bits & (1ULL << 12); }
  bool leech_seed() const { return bits & (1ULL << 13); }
  bool toxic() const { return bits & (1ULL << 14); }
  bool light_screen() const { return bits & (1ULL << 15); }
  bool reflect() const { return bits & (1ULL << 16); }
  bool transform() const { return bits & (1ULL << 17); }
  uint8_t confusion_left() const { return (bits >> 18) & 0b111; }
  void set_confusion_left(uint8_t val) {
    bits &= ~(uint64_t{0b111} << 18);
    bits |= (static_cast<uint64_t>(val) & 0b111) << 18;
  }
  uint8_t attacks() const { return (bits >> 21) & 0b111; }
  void set_attacks(uint8_t val) {
    bits &= ~(uint64_t{0b111} << 21);
    bits |= (static_cast<uint64_t>(val) & 0b111) << 21;
  }
  uint16_t state() const { return (bits >> 24) & 0xFFFF; }
  uint8_t substitute_hp() const { return (bits >> 40) & 0xFF; }
  uint8_t transform_species() const { return (bits >> 48) & 0xF; }
  uint8_t disable_left() const { return (bits >> 52) & 0xF; }
  void set_disable_left(uint8_t val) {
    bits &= ~(uint64_t{0xF} << 52);
    bits |= (static_cast<uint64_t>(val) & 0xF) << 52;
  }
  uint8_t disable_move() const { return (bits >> 56) & 0b111; }
  uint8_t toxic_counter() const { return (bits >> 59) & 0b11111; }
};

uint8_t encode_i4(int8_t x) { return static_cast<uint8_t>(x) & 0x0F; }

struct ActivePokemon {
  uint8_t bytes[32];

  Stats &stats() noexcept { return *reinterpret_cast<Stats *>(bytes + 0); }
  const Stats &stats() const noexcept {
    return *reinterpret_cast<const Stats *>(bytes + 0);
  }

  uint8_t boost_atk() const noexcept { return bytes[12] & 0b00001111; }
  uint8_t boost_def() const noexcept { return (bytes[12] & 0b11110000) >> 4; }
  uint8_t boost_spe() const noexcept { return bytes[13] & 0b00001111; }
  uint8_t boost_spc() const noexcept { return (bytes[13] & 0b11110000) >> 4; }
  uint8_t boost_acc() const noexcept { return bytes[14] & 0b00001111; }
  uint8_t boost_eva() const noexcept { return (bytes[14] & 0b11110000) >> 4; }

  void set_boost_atk(int8_t value) noexcept {
    bytes[12] = (bytes[12] & 0b11110000) | (encode_i4(value));
  }
  void set_boost_def(int8_t value) noexcept {
    bytes[12] = (bytes[12] & 0b00001111) | (encode_i4(value) << 4);
  }
  void set_boost_spe(int8_t value) noexcept {
    bytes[13] = (bytes[13] & 0b11110000) | (encode_i4(value));
  }
  void set_boost_spc(int8_t value) noexcept {
    bytes[13] = (bytes[13] & 0b00001111) | (encode_i4(value) << 4);
  }
  void set_boost_acc(int8_t value) noexcept {
    bytes[14] = (bytes[14] & 0b11110000) | (encode_i4(value));
  }
  void set_boost_eva(int8_t value) noexcept {
    bytes[14] = (bytes[14] & 0b00001111) | (encode_i4(value) << 4);
  }

  Volatiles &volatiles() noexcept {
    return *reinterpret_cast<Volatiles *>(bytes + 16);
  }
  const Volatiles &volatiles() const noexcept {
    return *reinterpret_cast<const Volatiles *>(bytes + 16);
  }
};

struct Side {
  uint8_t bytes[Sizes::Side];

  Pokemon &pokemon(size_t slot) noexcept {
    return *reinterpret_cast<Pokemon *>(bytes + slot * Sizes::Pokemon);
  }
  const Pokemon &pokemon(size_t slot) const noexcept {
    return *reinterpret_cast<const Pokemon *>(bytes + slot * Sizes::Pokemon);
  }

  ActivePokemon &active() noexcept {
    return *reinterpret_cast<ActivePokemon *>(bytes + Offsets::Side::active);
  }
  const ActivePokemon &active() const noexcept {
    return *reinterpret_cast<const ActivePokemon *>(bytes +
                                                    Offsets::Side::active);
  }

  std::array<uint8_t, 6> &order() noexcept {
    return *reinterpret_cast<std::array<uint8_t, 6> *>(bytes +
                                                       Offsets::Side::order);
  }
  const std::array<uint8_t, 6> &order() const noexcept {
    return *reinterpret_cast<const std::array<uint8_t, 6> *>(
        bytes + Offsets::Side::order);
  }

  uint8_t &order(size_t i) noexcept {
    return (bytes + Offsets::Side::order)[i];
  }
  const uint8_t &order(size_t i) const noexcept {
    return (bytes + Offsets::Side::order)[i];
  }
};

struct Battle {
  uint8_t bytes[PKMN_GEN1_BATTLE_SIZE];

  Side &side(size_t side) noexcept {
    return *reinterpret_cast<Side *>(bytes + side * Sizes::Side);
  }
  const Side &side(size_t side) const noexcept {
    return *reinterpret_cast<const Side *>(bytes + side * Sizes::Side);
  }
};

inline Battle &ref(pkmn_gen1_battle &battle) noexcept {
  return *reinterpret_cast<Battle *>(&battle);
}

inline const Battle &ref(const pkmn_gen1_battle &battle) noexcept {
  return *reinterpret_cast<const Battle *>(&battle);
}

struct Duration {
  uint32_t data = 0;

  uint8_t sleep(size_t slot) const noexcept {
    return (data >> (3 * slot)) & 0b111;
  }

  void set_sleep(size_t slot, uint8_t sleeps) noexcept {
    const uint32_t mask = 0b111 << (3 * slot);
    data = (data & ~mask) | ((sleeps & 0b111) << (3 * slot));
  }

  uint8_t confusion() const noexcept { return (data >> 18) & 0b111; }

  uint8_t disable() const noexcept { return (data >> 21) & 0b1111; }

  uint8_t attacking() const noexcept { return (data >> 25) & 0b111; }

  uint8_t binding() const noexcept { return (data >> 28) & 0b111; }
};

struct Durations {
  Duration d[2];

  Duration &duration(size_t i) noexcept { return d[i]; }

  const Duration &duration(size_t i) const noexcept { return d[i]; }
};

inline Durations &ref(pkmn_gen1_chance_durations &durations) noexcept {
  return *reinterpret_cast<Durations *>(&durations);
}

inline const Durations &
ref(const pkmn_gen1_chance_durations &durations) noexcept {
  return *reinterpret_cast<const Durations *>(&durations);
}

} // namespace View
