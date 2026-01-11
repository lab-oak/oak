#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/key.h>

namespace Hash {

void initialize(auto &device, auto &arr) {
  for (auto &x : arr) {
    x = device.uniform_64();
  }
}

struct PP {
  static constexpr int n_pp_buckets = 4;
  static constexpr int n_keys = std::pow(n_pp_buckets, 4);
  std::array<uint64_t, n_keys> hashes;
  PP() = default;
  PP(auto &device) { initialize(device, hashes); }

  uint8_t get_key(const std::array<PKMN::MoveSlot, 4> &moves) const noexcept {
    uint8_t key = 0;
    for (const auto &move_slot : moves) {
      key <<= 2;
      key ^= std::min(move_slot.pp, uint8_t{3});
    }
    return key;
  }

  uint64_t hash(const PKMN::Pokemon &pokemon) const noexcept {
    return hashes[get_key(pokemon.moves)];
  }
};

struct Status {
  static constexpr int n_status = Encode::Battle::Status::n_dim + 1;
  std::array<uint64_t, n_status> hashes;
  Status() = default;
  Status(auto &device) {
    for (auto &hash : hashes) {
      hash = device.uniform_64();
    }
  }
  uint64_t hash(const PKMN::Pokemon &pokemon,
                const uint8_t sleep) const noexcept {
    return hashes[Encode::Battle::Status::get_status_index(pokemon.status,
                                                           sleep) +
                  1];
  }
};

struct Slot {
  PP pp;
  Status status;
  Slot() = default;
  Slot(auto &device) : pp{device}, status{device} {}
  uint64_t hash(const PKMN::Pokemon &pokemon,
                const uint8_t sleep) const noexcept {
    return pp.hash(pokemon) ^ status.hash(pokemon, sleep);
  }
};

struct Stats {
  static constexpr int n_ratios = 13;
  std::array<uint64_t, n_ratios> atk;
  std::array<uint64_t, n_ratios> def;
  std::array<uint64_t, n_ratios> spe;
  std::array<uint64_t, n_ratios> spc;

  Stats() = default;
  Stats(auto &device) {
    initialize(device, atk);
    initialize(device, def);
    initialize(device, spe);
    initialize(device, spc);
  }

  static constexpr uint8_t ratio_key(uint16_t base, uint16_t cur) noexcept {
    const bool pos = (base <= cur);
    if (!pos) {
      std::swap(base, cur);
    }
    const auto halfs = std::min(6, 2 * ((cur - base) / base));
    return pos ? 6 + halfs : 6 - halfs;
  }

  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Pokemon &stored) const noexcept {
    return atk[ratio_key(stored.stats.atk, active.stats.atk)] ^
           def[ratio_key(stored.stats.def, active.stats.def)] ^
           spe[ratio_key(stored.stats.spe, active.stats.spe)] ^
           spc[ratio_key(stored.stats.spc, active.stats.spc)];
  }
};

static_assert(Stats::ratio_key(400, 100) == 0);
static_assert(Stats::ratio_key(100, 100) == 6);
static_assert(Stats::ratio_key(100, 400) == 12);

struct Species {
  std::array<uint64_t, static_cast<uint8_t>(PKMN::Species::Mew)> hashes;
  Species() = default;
  Species(auto &device) { initialize(device, hashes); }
  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Pokemon &stored) const noexcept {
    return hashes[static_cast<uint8_t>(active.species) - 1];
  }
};

struct Type {
  static constexpr int n_types =
      static_cast<uint8_t>(PKMN::Data::Type::Dragon) + 1;
  static constexpr int n_keys = n_types * n_types;
  std::array<uint64_t, n_keys> hashes;
  Type() = default;
  Type(auto &device) { initialize(device, hashes); }
  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Pokemon &stored) const noexcept {
    auto raw = static_cast<uint8_t>(active.types);
    return hashes[raw ^ (raw >> 4)];
  }
};

struct Boosts {
  static constexpr int n_boosts = 13;
  std::array<uint64_t, n_boosts> atk;
  std::array<uint64_t, n_boosts> def;
  std::array<uint64_t, n_boosts> spe;
  std::array<uint64_t, n_boosts> spc;
  Boosts() = default;
  Boosts(auto &device) {
    initialize(device, atk);
    initialize(device, def);
    initialize(device, spe);
    initialize(device, spc);
  }
  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Pokemon &stored) const noexcept {
    return atk[active.boosts.atk() + 6] ^ def[active.boosts.def() + 6] ^
           spe[active.boosts.spe() + 6] ^ spc[active.boosts.spc() + 6];
  }
};

struct Volatiles {
  Volatiles() = default;
  Volatiles(auto &device) {}
  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Duration &duration) const noexcept {
    return 0;
  }
};

struct Duration {};

struct ActivePokemon {
  Stats stats;
  Species species;
  Type types;
  Boosts boosts;
  Volatiles volatiles;
  Duration duration;
  ActivePokemon() = default;
  ActivePokemon(auto &device)
      : stats{device}, species{device}, types{device}, boosts{device},
        volatiles{device} {}

  uint64_t hash(const PKMN::ActivePokemon &active, const PKMN::Pokemon &pokemon,
                const PKMN::Duration &duration) const noexcept {
    return stats.hash(active, pokemon) ^ species.hash(active, pokemon) ^
           types.hash(active, pokemon) ^ boosts.hash(active, pokemon) ^
           volatiles.hash(active, duration);
  }
};

struct HP {
  static constexpr int n_buckets = 8;
  std::array<uint64_t, n_buckets> hashes;
  HP() = default;
  HP(auto &device) { initialize(device, hashes); }
  static constexpr uint8_t get_key(uint16_t base_hp, uint16_t hp) noexcept {
    return (8 * hp / base_hp) - (hp == base_hp);
  }

  uint64_t hash(const PKMN::Pokemon &pokemon) const noexcept {
    return hashes[get_key(pokemon.stats.hp, pokemon.hp)];
  }
};

static_assert(HP::get_key(800, 800) == 7);
static_assert(HP::get_key(800, 799) == 7);
static_assert(HP::get_key(800, 100) == 1);
static_assert(HP::get_key(800, 99) == 0);

struct Side {

  std::array<Slot, 6> slots;
  std::array<ActivePokemon, 6> actives;
  std::array<HP, 6> hps;
  Side() = default;
  Side(auto &device) {
    for (auto &slot : slots) {
      slot = Slot{device};
    }
    for (auto &active : actives) {
      active = ActivePokemon{device};
    }
  }

  uint64_t hash(const PKMN::Side &side,
                const PKMN::Duration &duration) const noexcept {
    uint64_t h = 0;
    const auto &stored = side.stored();
    if (stored.hp) {
      h ^=
          actives[side.order[0] - 1].hash(side.active, side.stored(), duration);
      h ^= hps[side.order[0] - 1].hash(stored);
    }
    for (auto slot = 2; slot <= 6; ++slot) {
      const auto id = side.order[slot - 1];
      if (id) {
        const auto &pokemon = side.pokemon[id - 1];
        const uint8_t sleep = duration.sleep(slot - 1);
        if (pokemon.hp) {
          h ^= slots[id - 1].hash(pokemon, sleep);
          h ^= hps[id - 1].hash(pokemon);
        }
      }
    }
    return h;
  }
};

struct Battle {
  std::array<Side, 2> sides;
  Battle() = default;
  Battle(auto &device) {
    for (auto &side : sides) {
      side = Side{device};
    }
  }
  uint64_t hash(const pkmn_gen1_battle &b,
                const pkmn_gen1_chance_durations &d) const noexcept {
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    return sides[0].hash(battle.sides[0], durations.get(0)) ^
           sides[1].hash(battle.sides[1], durations.get(1));
  }
};
}; // namespace Hash