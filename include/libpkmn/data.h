#pragma once

#include <pkmn.h>

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/layout.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace PKMN {

#pragma pack(push, 1)

struct alignas(1) Stats {
  uint16_t hp;
  uint16_t atk;
  uint16_t def;
  uint16_t spe;
  uint16_t spc;
  auto operator<=>(const Stats& other) const = default;
};

struct alignas(1) MoveSlot {
  Data::Move id;
  uint8_t pp;
  auto operator<=>(const MoveSlot& other) const = default;

};

static_assert(Data::Move::Substitute < Data::Move::Struggle);

struct alignas(1) Pokemon {
  Stats stats;
  std::array<MoveSlot, 4> moves;
  uint16_t hp;
  Data::Status status;
  Data::Species species;
  uint8_t types;
  uint8_t level;
  auto operator<=>(const Pokemon& other) const = default;


  int percent() const noexcept {
    return std::ceil(100 * static_cast<float>(hp) / stats.hp);
  }
};

struct alignas(1) Volatiles {
  uint64_t bits;

  auto operator<=>(const Volatiles& other) const = default;

  bool bide() const { return bits & (1ULL << 0); }
  void set_bide(bool val) { val ? bits |= (1ULL << 0) : bits &= ~(1ULL << 0); }
  bool thrashing() const { return bits & (1ULL << 1); }
  void set_thrashing(bool val) {
    val ? bits |= (1ULL << 1) : bits &= ~(1ULL << 1);
  }
  bool multi_hit() const { return bits & (1ULL << 2); }
  void set_multi_hit(bool val) {
    val ? bits |= (1ULL << 2) : bits &= ~(1ULL << 2);
  }
  bool flinch() const { return bits & (1ULL << 3); }
  void set_flinch(bool val) {
    val ? bits |= (1ULL << 3) : bits &= ~(1ULL << 3);
  }
  bool charging() const { return bits & (1ULL << 4); }
  void set_charging(bool val) {
    val ? bits |= (1ULL << 4) : bits &= ~(1ULL << 4);
  }
  bool binding() const { return bits & (1ULL << 5); }
  void set_binding(bool val) {
    val ? bits |= (1ULL << 5) : bits &= ~(1ULL << 5);
  }
  bool invulnerable() const { return bits & (1ULL << 6); }
  void set_invulnerable(bool val) {
    val ? bits |= (1ULL << 6) : bits &= ~(1ULL << 6);
  }
  bool confusion() const { return bits & (1ULL << 7); }
  void set_confusion(bool val) {
    val ? bits |= (1ULL << 7) : bits &= ~(1ULL << 7);
  }
  bool mist() const { return bits & (1ULL << 8); }
  void set_mist(bool val) { val ? bits |= (1ULL << 8) : bits &= ~(1ULL << 8); }
  bool focus_energy() const { return bits & (1ULL << 9); }
  void set_focus_energy(bool val) {
    val ? bits |= (1ULL << 9) : bits &= ~(1ULL << 9);
  }
  bool substitute() const { return bits & (1ULL << 10); }
  void set_substitute(bool val) {
    val ? bits |= (1ULL << 10) : bits &= ~(1ULL << 10);
  }
  bool recharging() const { return bits & (1ULL << 11); }
  void set_recharging(bool val) {
    val ? bits |= (1ULL << 11) : bits &= ~(1ULL << 11);
  }
  bool rage() const { return bits & (1ULL << 12); }
  void set_rage(bool val) {
    val ? bits |= (1ULL << 12) : bits &= ~(1ULL << 12);
  }
  bool leech_seed() const { return bits & (1ULL << 13); }
  void set_leech_seed(bool val) {
    val ? bits |= (1ULL << 13) : bits &= ~(1ULL << 13);
  }
  bool toxic() const { return bits & (1ULL << 14); }
  void set_toxic(bool val) {
    val ? bits |= (1ULL << 14) : bits &= ~(1ULL << 14);
  }
  bool light_screen() const { return bits & (1ULL << 15); }
  void set_light_screen(bool val) {
    val ? bits |= (1ULL << 15) : bits &= ~(1ULL << 15);
  }
  bool reflect() const { return bits & (1ULL << 16); }
  void set_reflect(bool val) {
    val ? bits |= (1ULL << 16) : bits &= ~(1ULL << 16);
  }
  bool transform() const { return bits & (1ULL << 17); }
  void set_transform(bool val) {
    val ? bits |= (1ULL << 17) : bits &= ~(1ULL << 17);
  }
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

constexpr uint8_t encode_i4(int8_t x) { return static_cast<uint8_t>(x) & 0x0F; }
constexpr int8_t decode_i4(uint8_t x) {
  x &= 0x0F;
  return static_cast<int8_t>((x ^ 0x08) - 0x08);
}

consteval bool i4_conversion_is_valid() {
  bool valid = true;
  for (int8_t b = -6; b <= 6; ++b) {
    valid &= decode_i4(encode_i4(b)) == b;
  }
  return valid;
}

static_assert(i4_conversion_is_valid());

struct alignas(1) Boosts {
  uint8_t bytes[4];

auto operator<=>(const Boosts &other) const = default;

  int8_t atk() const noexcept { return decode_i4(bytes[0] & 0b00001111); }
  int8_t def() const noexcept {
    return decode_i4((bytes[0] & 0b11110000) >> 4);
  }
  int8_t spe() const noexcept { return decode_i4(bytes[1] & 0b00001111); }
  int8_t spc() const noexcept {
    return decode_i4((bytes[1] & 0b11110000) >> 4);
  }
  int8_t acc() const noexcept { return decode_i4(bytes[2] & 0b00001111); }
  int8_t eva() const noexcept {
    return decode_i4((bytes[2] & 0b11110000) >> 4);
  }

  void set_atk(int8_t value) noexcept {
    bytes[0] = (bytes[0] & 0b11110000) | (encode_i4(value));
  }
  void set_def(int8_t value) noexcept {
    bytes[0] = (bytes[0] & 0b00001111) | (encode_i4(value) << 4);
  }
  void set_spe(int8_t value) noexcept {
    bytes[1] = (bytes[1] & 0b11110000) | (encode_i4(value));
  }
  void set_spc(int8_t value) noexcept {
    bytes[1] = (bytes[1] & 0b00001111) | (encode_i4(value) << 4);
  }
  void set_acc(int8_t value) noexcept {
    bytes[2] = (bytes[2] & 0b11110000) | (encode_i4(value));
  }
  void set_eva(int8_t value) noexcept {
    bytes[2] = (bytes[2] & 0b00001111) | (encode_i4(value) << 4);
  }
};

struct alignas(1) ActivePokemon {
  Stats stats;
  Data::Species species;
  uint8_t types;
  Boosts boosts;
  Volatiles volatiles;
  std::array<MoveSlot, 4> moves;

  auto operator<=>(const ActivePokemon& other) const = default;
};

struct alignas(1) Side {
  std::array<Pokemon, 6> pokemon;
  ActivePokemon active;
  std::array<uint8_t, 6> order;
  Data::Move last_selected_move;
  Data::Move last_used_move;

  Pokemon &get(auto slot) noexcept {
    assert(slot > 0 && slot <= 6);
    const auto id = order[slot - 1];
    assert(id > 0 && id <= 6);
    return pokemon[id - 1];
  }
  const Pokemon &get(auto slot) const noexcept {
    assert(slot > 0 && slot <= 6);
    const auto id = order[slot - 1];
    assert(id > 0 && id <= 6);
    return pokemon[id - 1];
  }

  Pokemon &stored() noexcept { return get(1); }
  const Pokemon &stored() const noexcept { return get(1); }
};

struct alignas(1) MoveDetails {
  uint8_t index;
  uint8_t counterable;
};

struct alignas(1) Battle {
  std::array<Side, 2> sides;
  uint16_t turn;
  uint16_t last_damage;
  std::array<MoveDetails, 2> last_moves;
  uint64_t rng;
};

struct alignas(1) Duration {
  uint32_t data;

  uint8_t sleep(auto slot) const noexcept {
    return (data >> (3 * slot)) & 0b111;
  }
  void set_sleep(auto slot, uint8_t sleeps) noexcept {
    const uint32_t mask = 0b111u << (3 * slot);
    data = (data & ~mask) | ((sleeps & 0b111u) << (3 * slot));
  }

  uint8_t confusion() const noexcept { return (data >> 18) & 0b111; }
  void set_confusion(uint8_t v) noexcept {
    const uint32_t mask = 0b111u << 18;
    data = (data & ~mask) | ((v & 0b111u) << 18);
  }

  uint8_t disable() const noexcept { return (data >> 21) & 0b1111; }
  void set_disable(uint8_t v) noexcept {
    const uint32_t mask = 0b1111u << 21;
    data = (data & ~mask) | ((v & 0b1111u) << 21);
  }

  uint8_t attacking() const noexcept { return (data >> 25) & 0b111; }
  void set_attacking(uint8_t v) noexcept {
    const uint32_t mask = 0b111u << 25;
    data = (data & ~mask) | ((v & 0b111u) << 25);
  }

  uint8_t binding() const noexcept { return (data >> 28) & 0b111; }
  void set_binding(uint8_t v) noexcept {
    const uint32_t mask = 0b111u << 28;
    data = (data & ~mask) | ((v & 0b111u) << 28);
  }
};

struct alignas(1) Durations {
  Duration durations[2];

  Duration &get(auto i) noexcept { return durations[i]; }
  const Duration &get(auto i) const noexcept { return durations[i]; }
};

#pragma pack(pop)

static_assert(sizeof(Battle) == Layout::Sizes::Battle);
static_assert(sizeof(Side) == Layout::Sizes::Side);
static_assert(sizeof(Pokemon) == Layout::Sizes::Pokemon);
static_assert(sizeof(Volatiles) == 8);
static_assert(sizeof(Stats) == 10);
static_assert(sizeof(MoveSlot) == 2);
static_assert(sizeof(Boosts) == 4);
static_assert(sizeof(ActivePokemon) == Layout::Sizes::ActivePokemon);
static_assert(sizeof(Durations) == Layout::Sizes::Durations);

inline PKMN::Battle &view(pkmn_gen1_battle &battle) noexcept {
  return *reinterpret_cast<PKMN::Battle *>(&battle);
}

inline const PKMN::Battle &view(const pkmn_gen1_battle &battle) noexcept {
  return *reinterpret_cast<const PKMN::Battle *>(&battle);
}

inline PKMN::Durations &view(pkmn_gen1_chance_durations &durations) noexcept {
  return *reinterpret_cast<PKMN::Durations *>(&durations);
}

inline const PKMN::Durations &
view(const pkmn_gen1_chance_durations &durations) noexcept {
  return *reinterpret_cast<const PKMN::Durations *>(&durations);
}

constexpr inline auto cast(const pkmn_gen1_battle &battle) noexcept {
  return std::bit_cast<PKMN::Battle>(battle);
}

constexpr inline auto switch_in(const Pokemon &pokemon) noexcept {
  ActivePokemon active{};
  active.stats = pokemon.stats;
  active.species = pokemon.species;
  active.types = pokemon.types;
  active.moves = pokemon.moves;
  return active;
}

} // namespace PKMN