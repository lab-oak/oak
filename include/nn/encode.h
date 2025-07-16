#pragma once

#include <battle/init.h>
#include <battle/view.h>
#include <data/legal-moves.h>
#include <data/status.h>

#include <array>
#include <cassert>

namespace Encode {

namespace Stats {

void write(const View::Stats &stats, float *t) {
  constexpr float max_stat_value = 999.0;
  // we use this same function for active but the stats come from ActivePokemon
  t[0] = stats.hp() / max_stat_value;
  t[1] = stats.atk() / max_stat_value;
  t[2] = stats.def() / max_stat_value;
  t[3] = stats.spe() / max_stat_value;
  t[4] = stats.spc() / max_stat_value;
}

} // namespace Stats

namespace Pokemon {

constexpr auto in_dim = 212;

constexpr auto n_status = 15;

constexpr auto get_status_index(auto status, uint8_t sleeps) {
  if (!static_cast<bool>(status)) {
    return 0;
  }
  // brn, par, psn(vol encodes tox), frz
  if (!Data::is_sleep(status)) {
    return std::countr_zero(static_cast<uint8_t>(status)) - 3;
  } else {
    if (!Data::self(status)) {
      return 4 + sleeps;
    } else {
      const auto s = static_cast<uint8_t>(status) & 7;
      return 12 + (s - 1);
    }
  }
}

static_assert(get_status_index(Data::Status::Poison, 0) == 0);
static_assert(get_status_index(Data::Status::Burn, 0) == 1);
static_assert(get_status_index(Data::Status::Freeze, 0) == 2);
static_assert(get_status_index(Data::Status::Paralysis, 0) == 3);
static_assert(get_status_index(Data::Status::Toxic, 0) == 0);
static_assert(get_status_index(Data::Status::Sleep7, 0) == 4);
static_assert(get_status_index(Data::Status::Sleep7, 1) == 5);
static_assert(get_status_index(Data::Status::Sleep6, 2) == 6);
static_assert(get_status_index(Data::Status::Sleep5, 3) == 7);
static_assert(get_status_index(Data::Status::Sleep4, 4) == 8);
static_assert(get_status_index(Data::Status::Sleep3, 5) == 9);
static_assert(get_status_index(Data::Status::Sleep2, 6) == 10);
static_assert(get_status_index(Data::Status::Sleep1, 7) == 11);
static_assert(get_status_index(Data::Status::Rest1, 2) == 12);
static_assert(get_status_index(Data::Status::Rest2, 1) == 13);
static_assert(get_status_index(Data::Status::Rest3, 0) == 14);

template <bool write_stats = true>
void write(const View::Pokemon &pokemon, auto sleep, float *t) {
  // Struggle and None do not have dimensions
  constexpr auto n_moves = static_cast<uint8_t>(Data::Move::Struggle) - 1;
  if constexpr (write_stats) {
    Stats::write(pokemon.stats(), t);
  }
  // one hot starting at index 5 for (id - 1) since None is not encoded
  for (const auto [id, pp] : pokemon.moves()) {
    if (id != Data::Move::Struggle && id != Data::Move::None) {
      t[(5 - 1) + static_cast<uint8_t>(id)] = static_cast<bool>(pp);
    }
  }
  if (static_cast<bool>(pokemon.status())) {
    t[5 + n_moves + get_status_index(pokemon.status(), sleep)] = 1;
  }
  t[5 + n_moves + n_status + (pokemon.types() & 16)] = 1;
  t[5 + n_moves + n_status + (pokemon.types() >> 4)] = 1;
}
} // namespace Pokemon

namespace Active {

constexpr auto in_dim = 198;

void write(const View::Volatiles &vol, const View::Duration &dur, float *t) {

  constexpr float chansey_sub = 706 / 4 + 1;

  // See data layout in extern/engine/src/lib/gen1/readme.md
  t[0] = vol.bide();
  t[1] = vol.thrashing();
  t[2] = vol.multi_hit();
  t[3] = vol.flinch();
  t[4] = vol.charging();
  t[5] = vol.binding();
  t[6] = vol.invulnerable();
  t[7] = vol.confusion();
  t[8] = vol.mist();
  t[9] = vol.focus_energy();
  t[10] = vol.substitute();
  t[11] = vol.recharging();
  t[12] = vol.rage();
  t[13] = vol.leech_seed();
  t[14] = vol.toxic();
  t[15] = vol.light_screen();
  t[16] = vol.reflect();
  t[17] = vol.transform();
  t[18] = vol.state() / 65535.0; // u16 bide damage? TODO
  t[19] = vol.substitute_hp() / chansey_sub;
  t[21] = vol.toxic_counter() / 16.0;
}

void write(const View::Pokemon &pokemon, const View::ActivePokemon &active,
           const View::Duration &dur, float *t) {}
} // namespace Active

namespace Team {

consteval auto get_species_move_list_size() {
  uint16_t size = 0;
  for (auto s = 1; s <= 149; ++s) {
    for (auto m = 0; m < 166; ++m) {
      if (Data::MOVE_POOLS[s][m] || !m) {
        ++size;
      }
    }
  }
  return size;
}

constexpr auto species_move_list_size = get_species_move_list_size();

constexpr auto in_dim = species_move_list_size;
constexpr auto out_dim = in_dim;

consteval auto get_species_move_data() {
  std::array<std::array<uint16_t, 166>, 152> table{};
  std::array<std::pair<uint8_t, uint8_t>, species_move_list_size> list{};

  uint16_t index = 0;
  for (auto s = 1; s <= 149; ++s) {
    for (auto m = 0; m < 166; ++m) {
      if (Data::MOVE_POOLS[s][m] || !m) {
        table[s][m] = index;
        list[index] = {s, m};
        ++index;
      }
    }
  }
  return std::pair<decltype(table), decltype(list)>{table, list};
}

constexpr auto SPECIES_MOVE_DATA = get_species_move_data();
constexpr auto SPECIES_MOVE_TABLE = SPECIES_MOVE_DATA.first;
constexpr auto SPECIES_MOVE_LIST = SPECIES_MOVE_DATA.second;

inline constexpr auto species_move_table(const auto species, const auto move) {
  return SPECIES_MOVE_TABLE[static_cast<uint8_t>(species)]
                           [static_cast<uint8_t>(move)];
}

inline constexpr auto species_move_list(const auto index) {
  return SPECIES_MOVE_LIST[index];
}

void write(const Init::Team &team, float *const t) {
  for (const auto &set : team) {
    if (static_cast<bool>(set.species)) {
      t[species_move_table(set.species, 0)] = 1;
    }
    for (const auto move : set.moves) {
      if (static_cast<bool>(move)) {
        t[species_move_table(set.species, move)] = 1;
      }
    }
  }
}

[[nodiscard]] bool write_policy_mask(const Init::Team &team, float *const t) {
  bool needs_species = false;
  bool complete = true;
  for (const auto &set : team) {
    if (static_cast<bool>(set.species)) {
      bool needs_move = false;
      auto n_moves = 0;
      for (const auto move : set.moves) {
        n_moves += static_cast<bool>(move);
      }
      if (n_moves < std::min(Data::move_pool_size(set.species), (uint8_t)4)) {
        complete = false;
        for (const auto move : move_pool(set.species)) {
          if (!static_cast<bool>(move)) {
            break;
          }
          t[species_move_table(set.species, move)] = 1.0;
        }
        for (const auto move : set.moves) {
          if (static_cast<bool>(move)) {
            t[species_move_table(set.species, move)] = 0.0;
          }
        }
      }
    } else {
      needs_species = true;
      complete = false;
    }
  }
  if (needs_species) {
    for (int i = 1; i <= 149; ++i) {
      t[species_move_table(i, 0)] = 1.0;
    }
    for (const auto &set : team) {
      if (static_cast<bool>(set.species)) {
        t[species_move_table(set.species, 0)] = 0.0;
      }
    }
  }
  if (complete) {
    for (const auto &set : team) {
      t[species_move_table(set.species, 0)] = 1.0;
    }
  }

  return complete;
}

void apply_index_to_team(Init::Team &team, auto s, auto m) {
  if (!m) {
    for (auto &set : team) {
      if (set.species == Data::Species::None) {
        set.species = static_cast<Data::Species>(s);
        return;
      } else if (set.species == static_cast<Data::Species>(s)) {
        std::swap(set, team[0]);
        return;
      }
    }
    // assert(false, "Cant add species to set/set active");
    assert(false);
  } else {
    for (auto &set : team) {
      if (set.species == static_cast<Data::Species>(s)) {
        for (auto &move : set.moves) {
          if (move == Data::Move::None) {
            move = static_cast<Data::Move>(m);
            return;
          }
        }
        // assert(false, "Cant add move to species");
        assert(false);
      }
    }
  }
}

} // namespace Team

}; // namespace Encode

// We know the encoding is correct if it is reversible
namespace Decode {};