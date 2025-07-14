#pragma once

#include <battle/init.h>
#include <battle/view.h>
#include <data/legal-moves.h>
#include <data/status.h>

#include <array>
#include <cassert>

namespace Encode {

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
           const View::Duration &dur) {}
} // namespace Active

namespace Pokemon {

constexpr auto in_dim = 212;

constexpr auto n_status = 14;

constexpr auto get_status_index(uint8_t status, uint8_t sleeps) {
  if (!status) {
    return 0;
  }
  auto index = 0;
  if (!Data::is_sleep(status)) {
    index = std::countr_zero(status) - 4;
    assert((index >= 0) && (index < 4));
  } else {
    if (!Data::self(status)) {
      index = 4 + sleeps;
      assert((index >= 4) && (index < 12));
    } else {
      const auto s = status & 7;
      index = 12 + (s - 1);
      assert((index >= 12) && (index < 14));
    }
  }
  return index + 1;
}

template <bool write_stats = true>
void write(const View::Pokemon &pokemon, auto sleep, float *t) {
  constexpr auto n_moves =
      static_cast<uint8_t>(Data::Move::Struggle) - 1; // no None dim
  constexpr float max_stat_value = 999.0;
  // we use this same function for active but the stats come from ActivePokemon
  if constexpr (write_stats) {
    t[0] = pokemon.stats().hp() / max_stat_value;
    t[1] = pokemon.stats().atk() / max_stat_value;
    t[2] = pokemon.stats().def() / max_stat_value;
    t[3] = pokemon.stats().spe() / max_stat_value;
    t[4] = pokemon.stats().spc() / max_stat_value;
  }
  // one hot starting at index 5 for (id - 1) since None is not encoded
  for (const auto [id, pp] : pokemon.moves()) {
    t[4 + static_cast<uint8_t>(id)] =
        static_cast<bool>(id) && static_cast<bool>(pp);
  }
  if (static_cast<bool>(pokemon.status())) {
    t[5 + n_moves + get_status_index(pokemon.status(), sleep)] = 1;
  }
  t[5 + n_moves + n_status + (pokemon.types() & 16)] = 1;
  t[5 + n_moves + n_status + (pokemon.types() >> 4)] = 1;
}
} // namespace Pokemon

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