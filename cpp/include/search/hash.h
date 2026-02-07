#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/key.h>

namespace Hash {

void initialize(auto &device, auto &arr) {
  for (auto &x : arr) {
    x = device.uniform_64();
  }
}

struct Pokemon {

  struct HP {
    using Key = uint8_t;
    static constexpr int n_buckets = 8;
    std::array<uint64_t, n_buckets> hashes;
    HP() = default;
    HP(auto &device) { initialize(device, hashes); }
    // TODO finish so that battles return a collision-less key
    // to be used with std::map
    static constexpr Key get_key(uint16_t base_hp, uint16_t hp) noexcept {
      return (n_buckets * hp / base_hp) - (hp == base_hp);
    }

    uint64_t hash(const PKMN::Pokemon &pokemon) const noexcept {
      return hashes[get_key(pokemon.stats.hp, pokemon.hp)];
    }
  };

  // static_assert(HP::get_key(800, 800) == 7);
  // static_assert(HP::get_key(800, 799) == 7);
  // static_assert(HP::get_key(800, 100) == 1);
  // static_assert(HP::get_key(800, 99) == 0);

  struct PP {
    static constexpr int n_pp_buckets = 4;
    static constexpr int n_keys =
        n_pp_buckets * n_pp_buckets * n_pp_buckets * n_pp_buckets;
    std::array<uint64_t, n_keys> hashes;
    PP() = default;
    PP(auto &device) { initialize(device, hashes); }

    static constexpr uint8_t
    get_key(const std::array<PKMN::MoveSlot, 4> &moves) noexcept {
      uint8_t key = 0;
      for (const auto &move_Pokemon : moves) {
        key *= n_pp_buckets;
        key += std::min(move_Pokemon.pp, uint8_t{n_pp_buckets - 1});
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
      const auto status_index = (pokemon.status == PKMN::Data::Status::None)
                                    ? 0
                                    : Encode::Battle::Status::get_status_index(
                                          pokemon.status, sleep) +
                                          1;
      return hashes[status_index];
    }
  };

  HP hp;
  PP pp;
  Status status;
  Pokemon() = default;
  Pokemon(auto &device) : hp{device}, pp{device}, status{device} {}
  uint64_t hash(const PKMN::Pokemon &pokemon,
                const uint8_t sleep) const noexcept {
    if (pokemon.hp == 0) {
      return 0;
    }
    return hp.hash(pokemon) ^ pp.hash(pokemon) ^ status.hash(pokemon, sleep);
  }
};

struct ActivePokemon {
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
      return hashes[(raw & 0x0F) + 15 * (raw >> 4)];
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

  struct Duration {
    static constexpr int max_duration = 6;
    std::array<uint64_t, max_duration> confusion;
    std::array<uint64_t, max_duration> disable;
    std::array<uint64_t, max_duration> attacking;
    std::array<uint64_t, max_duration> binding;
    Duration() = default;
    Duration(auto &device) {
      initialize(device, confusion);
      initialize(device, disable);
      initialize(device, attacking);
      initialize(device, binding);
    }
    uint64_t hash(const PKMN::Duration &duration) const noexcept {
      return confusion[duration.confusion()] ^ disable[duration.disable()] ^
             attacking[duration.attacking()] ^ binding[duration.binding()];
    }
  };

  struct Volatiles {
    std::array<uint64_t, 2> recharging;
    std::array<uint64_t, 2> reflect;
    std::array<uint64_t, 2> light_screen;
    Volatiles() = default;
    Volatiles(auto &device) {
      initialize(device, recharging);
      initialize(device, reflect);
      initialize(device, light_screen);
    }
    uint64_t hash(const PKMN::ActivePokemon &active,
                  const PKMN::Pokemon &stored) const noexcept {
      const auto &vol = active.volatiles;
      return recharging[vol.recharging()] ^ reflect[vol.reflect()] ^
             light_screen[vol.light_screen()];
    }
  };

  Stats stats;
  Species species;
  Type types;
  Boosts boosts;
  Volatiles volatiles;
  Duration duration;
  ActivePokemon() = default;
  ActivePokemon(auto &device)
      : stats{device}, species{device}, types{device}, boosts{device},
        volatiles{device}, duration{device} {}

  uint64_t hash(const PKMN::ActivePokemon &active, const PKMN::Pokemon &pokemon,
                const PKMN::Duration &dur) const noexcept {
    return stats.hash(active, pokemon) ^ species.hash(active, pokemon) ^
           types.hash(active, pokemon) ^ boosts.hash(active, pokemon) ^
           volatiles.hash(active, pokemon) ^ duration.hash(dur);
  }
};

struct Side {

  struct State {
    uint64_t last;
    uint64_t active;
    std::array<uint64_t, 6> pokemon;
  };

  State state;
  std::array<Pokemon, 6> pokemon;
  std::array<ActivePokemon, 6> actives;

  Side() = default;
  Side(auto &device) : state{} {
    for (auto &p : pokemon) {
      p = Pokemon{device};
    }
    for (auto &active : actives) {
      active = ActivePokemon{device};
    }
  }

  void hash_active(const PKMN::Side &side,
                   const PKMN::Duration &duration) noexcept {
    const auto &stored = side.stored();
    const auto id = side.order[0];
    if (stored.hp) {
      state.active = actives[id - 1].hash(side.active, stored, duration);
      const uint8_t sleep = duration.sleep(0);
      state.pokemon[id - 1] = pokemon[id - 1].hash(stored, sleep);
    } else {
      state.active = 0;
      state.pokemon[id - 1] = 0;
    }
  }

  void init(const PKMN::Side &side, const PKMN::Duration &duration) noexcept {
    state = {};
    hash_active(side, duration);
    state.last ^= state.active;
    state.last ^= state.pokemon[side.order[0] - 1];

    for (auto slot = 2; slot <= 6; ++slot) {
      const auto id = side.order[slot - 1];
      if (id) {
        const auto &p = side.pokemon[id - 1];
        const uint8_t sleep = duration.sleep(slot - 1);
        if (p.hp) {
          const uint64_t pokemon_hash = pokemon[id - 1].hash(p, sleep);
          state.pokemon[id - 1] = pokemon_hash;
          state.last ^= pokemon_hash;
        } else {
          state.pokemon[id - 1] = 0;
        }
      }
    }
  }

  // TODO remove this?
  void update(const PKMN::Side &updated_side,
              const PKMN::Duration &updated_duration,
              pkmn_choice choice) noexcept {
    const auto choice_type = choice & 3;
    const auto choice_data = choice >> 2;
    switch (choice_type) {
    case 1: {
      assert(choice_data >= 0 && choice_data <= 4);
      _update(updated_side, updated_duration);
      return;
    }
    case 2: {
      assert(choice_data >= 2 && choice_data <= 6);
      _update(updated_side, updated_duration);
      return;
    }
    case 0: {
      // TODO is this necessary? I forgot.
      _update(updated_side, updated_duration);
      return;
    }
    default: {
      assert(false);
      return;
    }
    }
  }

  void print() const noexcept {
    std::cout << "a: " << state.active << '\n';
    for (auto id = 1; id <= 6; ++id) {
      std::cout << id << ": " << state.pokemon[id - 1] << '\n';
    }
  }

  void _update(const PKMN::Side &updated_side,
               const PKMN::Duration &updated_duration) noexcept {
    const auto id = updated_side.order[0];
    // undo
    state.last ^= state.active;
    state.last ^= state.pokemon[id - 1];
    // update
    hash_active(updated_side, updated_duration);
    // apply updated
    state.last ^= state.active;
    state.last ^= state.pokemon[id - 1];
  }
};

struct State {
  Side::State s1;
  Side::State s2;
};

struct Battle {
  std::array<Side, 2> sides;
  Battle() = default;
  Battle(auto &device) {
    for (auto &side : sides) {
      side = Side{device};
    }
  }
  void print() const {
    for (auto &side : sides) {
      side.print();
    }
  }
  void init(const pkmn_gen1_battle &b,
            const pkmn_gen1_chance_durations &d) noexcept {
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    sides[0].init(battle.sides[0], durations.get(0));
    sides[1].init(battle.sides[1], durations.get(1));
  }
  uint64_t last() const noexcept {
    return sides[0].state.last ^ sides[1].state.last;
  }
  State state() const noexcept { return {sides[0].state, sides[1].state}; }
  void set(const State &state) noexcept {
    sides[0].state = state.s1;
    sides[1].state = state.s2;
  }

  void update(const pkmn_gen1_battle &b, const pkmn_gen1_chance_durations &d,
              const pkmn_choice c1, const pkmn_choice c2) noexcept {
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    sides[0].update(battle.sides[0], durations.get(0), c1);
    sides[1].update(battle.sides[1], durations.get(1), c2);
  }
};

static_assert(Pokemon::PP::get_key(std::array<PKMN::MoveSlot, 4>{}) == 0);
static_assert(Pokemon::PP::get_key(std::array<PKMN::MoveSlot, 4>{
                  PKMN::MoveSlot{PKMN::Data::Move::None, 61},
                  PKMN::MoveSlot{PKMN::Data::Move::None, 61},
                  PKMN::MoveSlot{PKMN::Data::Move::None, 61},
                  PKMN::MoveSlot{PKMN::Data::Move::None, 61}}) ==
              (Pokemon::PP::n_keys - 1));

static_assert(ActivePokemon::Stats::ratio_key(400, 100) == 0);
static_assert(ActivePokemon::Stats::ratio_key(100, 100) == 6);
static_assert(ActivePokemon::Stats::ratio_key(100, 400) == 12);

}; // namespace Hash