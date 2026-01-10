#pragma once

#include <encode/battle/key.h>

namespace Hash {

void initialize(auto &device, auto &arr) {
  for (auto &x : arr) {
    x = device.uniform_64();
  }
}

struct PP {
  static constexpr auto n_pp_buckets = 4;
  static constexpr auto n_keys = std::pow(n_pp_buckets, 4);
  std::array<n_keys, uint64_t> hashes;

  PP(auto &device) {
    for (auto &hash : hashes) {
      hash = device.uniform_64();
    }
  }

  uint8_t get_key(const std::array<PKMN::MoveSlot, 4> &moves) const noexcept {
    uint8_t key = 0;
    for (const auto &move_slot : moves) {
      key <<= 2;
      key ^= std::min(move_slot.pp, 3);
    }
    return key;
  }

  uint64_t hash(const PKMN::Pokemon &pokemon) const noexcept {
    return hashes[get_key(pokemon.moves)];
  }
};

struct Status {
  static constexpr auto n_status = Encode::Battle::Status::n_dim + 1;
  std::array<uint64_t, n_status> hashes;
  Status(auto &device) {
    for (auto &hash : hashes) {
      hash = device.uniform_64();
    }
    uint64_t hash(const PKMN::Pokemon &pokemon, const auto sleep)
        const noexcept {
      return hashes[Status::get_status_index(pokemon.status, sleep) + 1];
    }
  }
};

struct Slot {
  PP pp;
  Status status;
  Slot(auto &device) : pp{device}, status{device} {}
  uint64_t hash(const PKMN::Pokemon &pokemon, const auto sleep) const noexcept {
    return pp.hash(pokemon) ^ status.hash(pokemon, sleep);
  }
};

struct Stats {
  static constexpr auto n_ratios = 13;
  std::array<n_ratios, uint64_t> atk;
  std::array<n_ratios, uint64_t> def;
  std::array<n_ratios, uint64_t> spe;
  std::array<n_ratios, uint64_t> spc;

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
    return half + 6 * pos;
  }

  static_assert(ratio_key(100, 100) == 6);
  static_assert(ratio_key(400, 100) == 12);
  static_assert(ratio_key(100, 400) == 0);

  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Pokemon &stored) const noexcept {
    return atk[ratio_key(stored.stats.atk, active.stats.atk)] ^
           def[ratio_key(stored.stats.def, active.stats.dep)] ^
           spe[ratio_key(stored.stats.spe, active.stats.spe)] ^
           spc[ratio_key(stored.stats.spc, active.stats.spc)];
  }
};

struct Species {
  std::array<uint64_t static_cast<uint8_t>(PKMN::Species::Mew)> hashes;
  Species(auto &device) { initialize(device, hashes); }
  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Pokemon &stored) const noexcept {
    return hashes[static_cast<uint8_t>(active.species) - 1];
  }
};

struct Types {
  static constexpr auto n_types = static_cast<uint8_t>(PKMN::Types::Dragon) - 1;
  std::array<uint64_t, std::pow(n_types, 2)> hashes;
  Types(auto &device) { initialize(device, hashes); }
  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Pokemon &stored) const noexcept {
    auto raw = static_cast<uint8_t>(active.types);
    return hashes[raw ^ (raw >> 4)];
  }
};

struct Boosts {
  static constexpr auto n_boosts = 13;
  std::array<uint64_t, n_boosts> atk;
  std::array<uint64_t, n_boosts> def;
  std::array<uint64_t, n_boosts> spe;
  std::array<uint64_t, n_boosts> spc;
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
  Volatiles(auto &device) {}
  uint64_t hash(const PKMN::ActivePokemon &active,
                const PKMN::Duration &duration) const noexcept {
    return 0;
  }
};

struct ActivePokemon {
  Stats stats;
  Species species;
  Types types;
  Boosts boosts;
  Volatiles volatiles;

  ActivePokemon(auto &device)
      : stats{device}, species{device}, types{device}, boosts{device},
        volatiles{device} {}

  uint64_t hash(const PKMN::ActivePokemon &active) const noexcept {
    return stats.hash(active) ^ species.hash(active) ^ types.hash(active) ^
           boosts.hash(active) ^ Volatiles.hash(active);
  }
};

struct Side {
  std::array<Slot, 6> slots;
  std::array<ActivePokemon, 6> actives;

  Side(auto &device) {
    for (auto &slot : slots) {
      slot = Slot{device};
    }
    for (auto &active : active) {
      active = Active{device};
    }
  }

  uint64_t hash(const PKMN::Side &side,
                const PKMN::Duration &duration) const noexcept {
    uint64_t h = actives.hash(side.active, side.stored(), duration);
    for (auto slot = 2; slot <= 6; ++slot) {
      h ^= 0;
    }
  }
};

struct Battle {
  std::array<Side, 2> sides;
  Battle(auto &device) {
    for (auto &side : sides) {
      side = Side{device};
    }
  }
  uint64_t hash(const PKMN::Battle &battle) const noexcept {
    return sides[0].hash(battle.sides[0]) ^ sides[1].hash(battle.sides[1]);
  }
}
}; // namespace Hash