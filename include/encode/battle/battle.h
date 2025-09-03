#pragma once

#include <libpkmn/data.h>
#include <libpkmn/data/status.h>
#include <libpkmn/pkmn.h>

#include <array>
#include <cassert>

namespace Encode {

namespace Battle {

using PKMN::Data::Move;
using PKMN::Data::Species;
using PKMN::Data::Type;

namespace Stats {
constexpr float max_stat_value = 999;
constexpr float max_hp_value = 706;
constexpr auto n_dim = 5;
float *write(const PKMN::Stats &stats, float *t) {
  t[0] = stats.hp / max_hp_value;
  t[1] = stats.atk / max_stat_value;
  t[2] = stats.def / max_stat_value;
  t[3] = stats.spe / max_stat_value;
  t[4] = stats.spc / max_stat_value;
  return t + n_dim;
}

consteval auto dim_labels() {
  return std::array<std::array<char, 4>, n_dim>{
      {"HP", "ATK", "DEF", "SPE", "SPC"}};
}
} // namespace Stats

namespace MoveSlots {
constexpr auto n_dim = static_cast<uint8_t>(Move::Struggle) - 1;
float *write(const std::array<PKMN::MoveSlot, 4> &move_slots, float *t) {
  for (const auto [id, pp] : move_slots) {
    if (id != Move::Struggle && id != Move::None) {
      t[static_cast<uint8_t>(id) - 1] = static_cast<bool>(pp);
    }
  }
  return t + n_dim;
}

consteval auto dim_labels() {
  std::array<std::array<char, 13>, n_dim> result{};
  for (auto i = 0; i < MoveSlots::n_dim; ++i) {
    result[i] = PKMN::Data::MOVE_CHAR_ARRAY[i + 1];
  }
  return result;
}
} // namespace MoveSlots

namespace Status {
constexpr auto get_status_index(auto status, uint8_t sleeps) {
  // brn, par, psn(vol encodes tox), frz
  assert(static_cast<bool>(status));
  if (!is_sleep(status)) {
    const auto n = std::countr_zero(static_cast<uint8_t>(status)) - 3;
    assert(n >= 0 && n <= 3);
    return n;
  } else {
    if (!self(status)) {
      assert(sleeps > 0);
      return 3 + sleeps;
    } else {
      const auto s = static_cast<uint8_t>(status) & 7;
      assert(s > 0 && s <= 3);
      return 14 - s;
    }
  }
}

static_assert(get_status_index(PKMN::Data::Status::Poison, 0) == 0);
static_assert(get_status_index(PKMN::Data::Status::Burn, 0) == 1);
static_assert(get_status_index(PKMN::Data::Status::Freeze, 0) == 2);
static_assert(get_status_index(PKMN::Data::Status::Paralysis, 0) == 3);
static_assert(get_status_index(PKMN::Data::Status::Toxic, 0) == 0);
// further down corresponds to more likely to wake up
static_assert(get_status_index(PKMN::Data::Status::Sleep7, 1) == 4);
static_assert(get_status_index(PKMN::Data::Status::Sleep6, 2) == 5);
static_assert(get_status_index(PKMN::Data::Status::Sleep5, 3) == 6);
static_assert(get_status_index(PKMN::Data::Status::Sleep4, 4) == 7);
static_assert(get_status_index(PKMN::Data::Status::Sleep3, 5) == 8);
static_assert(get_status_index(PKMN::Data::Status::Sleep2, 6) == 9);
static_assert(get_status_index(PKMN::Data::Status::Sleep1, 7) == 10);
static_assert(get_status_index(PKMN::Data::Status::Rest3, 1) == 11);
static_assert(get_status_index(PKMN::Data::Status::Rest2, 2) == 12);
static_assert(get_status_index(PKMN::Data::Status::Rest1, 3) == 13);

// toxic and poison get the same status index, toxic also has a vol flag and a
// counter
constexpr auto n_dim = 14;

float *write(const auto status, const auto sleep, float *t) {
  if (static_cast<bool>(status)) {
    t[get_status_index(status, sleep)] = 1;
  }
  return t + n_dim;
}

consteval auto dim_labels() {
  return std::array<std::array<char, 5>, n_dim>{
      {"PSN", "BRN", "FRZ", "PAR", "SLP1", "SLP2", "SLP3", "SLP4", "SLP5",
       "SLP6", "SLP7", "RST1", "RST2", "RST3"}};
}
} // namespace Status

namespace Types {
constexpr auto n_dim = 15;
float *write(const auto types, float *t) {
  t[types % 16] = 1;
  t[types / 16] = 1;
  return t + n_dim;
}
} // namespace Types

namespace Pokemon {
constexpr auto n_dim =
    Stats::n_dim + MoveSlots::n_dim + Status::n_dim + Types::n_dim;

void write(const PKMN::Pokemon &pokemon, auto sleep, float *t) {
  t = Stats::write(pokemon.stats, t);
  t = MoveSlots::write(pokemon.moves, t);
  t = Status::write(pokemon.status, sleep, t);
  t = Types::write(pokemon.types, t);
}

consteval auto get_dim_labels() {
  std::array<std::array<char, 13>, n_dim> result{};
  const auto copy = [](const auto &src, auto &dest) {
    for (auto i = 0; i < src.size(); ++i) {
      dest[i] = src[i];
    }
  };
  auto index = 0;
  for (auto i = 0; i < Stats::n_dim; ++i) {
    copy(Stats::dim_labels()[i], result[index + i]);
  }
  index += Stats::n_dim;
  for (auto i = 0; i < MoveSlots::n_dim; ++i) {
    copy(MoveSlots::dim_labels()[i], result[index + i]);
  }
  index += MoveSlots::n_dim;
  for (auto i = 0; i < Status::n_dim; ++i) {
    copy(Status::dim_labels()[i], result[index + i]);
  }
  index += Status::n_dim;
  for (auto i = 0; i < Types::n_dim; ++i) {
    copy(PKMN::Data::TYPE_CHAR_ARRAY[i], result[index + i]);
  }
  return result;
}

constexpr auto dim_labels = get_dim_labels();

} // namespace Pokemon

namespace Boosts {

constexpr auto n_dim = 6;
constexpr float scale = 1 / 4.0;
constexpr float scale_acceva = 1 / 3.0;
float *write(const PKMN::ActivePokemon &active, float *t) {
  const auto get_multiplier = [](auto index) -> float {
    const auto x = PKMN::Data::boosts[index + 6];
    return (float)x[0] / x[1];
  };

  t[0] = get_multiplier(active.boosts.atk()) * scale;
  t[1] = get_multiplier(active.boosts.def()) * scale;
  t[2] = get_multiplier(active.boosts.spe()) * scale;
  t[3] = get_multiplier(active.boosts.spc()) * scale;
  t[4] = get_multiplier(active.boosts.acc()) * scale_acceva;
  t[5] = get_multiplier(active.boosts.eva()) * scale_acceva;
  return t + n_dim;
}

consteval auto dim_labels() {
  return std::array<std::array<char, 4>, n_dim>{
      {"atk", "def", "spe", "spc", "acc", "eva"}};
}
} // namespace Boosts

namespace Volatiles {
constexpr auto n_dim = 19;
float *write(const PKMN::Volatiles &vol, float *t) {

  constexpr float chansey_sub = 706 / 4 + 1;

  // See data layout in extern/engine/src/lib/gen1/readme.md
  // hidden data is replaced with normalized durations
  t[0] = vol.bide();
  t[1] = vol.thrashing();
  t[2] = vol.charging();
  t[3] = vol.binding();
  t[4] = vol.invulnerable();
  t[5] = vol.confusion();
  t[6] = vol.mist();
  t[7] = vol.focus_energy();
  t[8] = vol.substitute();
  t[9] = vol.recharging();
  t[10] = vol.rage();
  t[11] = vol.leech_seed();
  t[12] = vol.toxic();
  t[13] = vol.light_screen();
  t[14] = vol.reflect();
  t[15] = vol.transform();
  // confusion_left
  // attacks (thrashing/binding) left
  t[16] = vol.state() / (float)std::numeric_limits<uint16_t>::max();
  t[17] = vol.substitute_hp() / chansey_sub;
  // transform id
  // disable left
  // disable move slot; just zero out the move encoding
  t[18] = vol.toxic_counter() / 16.0;
  return t + n_dim;
}

consteval auto dim_labels() {
  return std::array<std::array<char, 13>, n_dim>{
      {"bide", "thrashing", "charging", "binding", "invulner", "confusion",
       "mist", "focus_energy", "substitute", "recharging", "rage", "leech_seed",
       "toxic", "light_screen", "reflect", "transform", "state", "sub_hp"}};
}
} // namespace Volatiles

namespace Duration {
constexpr auto n_confusion = 5;
constexpr auto n_disable = 8;
constexpr auto n_attacking = 3; // bide = thrashing
constexpr auto n_binding = 4;
constexpr auto n_dim = n_confusion + n_disable + n_attacking + n_binding;

float *write(const PKMN::Duration &duration, float *t) {
  if (const auto confusion = duration.confusion()) {
    assert(confusion <= n_confusion);
    t[confusion - 1] = 1;
  }
  t += n_confusion;

  if (const auto disable = duration.disable()) {
    assert(disable <= n_disable);
    t[disable - 1] = 1;
  }
  t += n_disable;

  if (const auto attacking = duration.attacking()) {
    assert(attacking <= n_attacking);
    t[attacking - 1] = 1;
  }
  t += n_attacking;

  if (const auto binding = duration.binding()) {
    assert(binding <= n_binding);
    t[binding - 1] = 1;
  }
  t += n_binding;
  return t;
}

consteval auto dim_labels() {
  std::array<std::array<char, 10>, n_dim> result;

  auto index = 0;
  for (auto i = 0; i < n_confusion; ++i) {
    result[index + i] = {"confusion"};
    result[index + i][9] = static_cast<char>('1' + i);
  }
  index += n_confusion;

  for (auto i = 0; i < n_disable; ++i) {
    result[index + i] = {"disable"};
    result[index + i][7] = static_cast<char>('1' + i);
  }
  index += n_disable;

  for (auto i = 0; i < n_attacking; ++i) {
    result[index + i] = {"attacking"};
    result[index + i][9] = static_cast<char>('1' + i);
  }
  index += n_attacking;

  for (auto i = 0; i < n_binding; ++i) {
    result[index + i] = {"binding"};
    result[index + i][7] = static_cast<char>('1' + i);
  }
  return result;
}
} // namespace Duration

namespace Active {

constexpr auto n_dim = Stats::n_dim + Types::n_dim + Boosts::n_dim +
                       Volatiles::n_dim + MoveSlots::n_dim + Duration::n_dim +
                       Pokemon::n_dim;

void write(const PKMN::Pokemon &pokemon, const PKMN::ActivePokemon &active,
           const PKMN::Duration &duration, float *t) {
  t = Stats::write(active.stats, t);
  t = Types::write(active.types, t);
  t = Boosts::write(active, t);
  t = Volatiles::write(active.volatiles, t);
  t = MoveSlots::write(active.moves, t);
  // disable
  if (const auto slot = active.volatiles.disable_move()) {
    assert(slot != 0);
    t[static_cast<uint8_t>(active.moves[slot].id)] = 0; // TODO
  }
  t = Duration::write(duration, t);
  Pokemon::write(pokemon, duration.sleep(0), t);
}

consteval auto get_dim_labels() {
  std::array<std::array<char, 13>, n_dim> result{};

  const auto copy = [](const auto &src, auto &dest) {
    for (auto i = 0; i < src.size(); ++i) {
      dest[i] = src[i];
    }
  };

  auto index = 0;
  for (auto i = 0; i < Stats::n_dim; ++i) {
    copy(Stats::dim_labels()[i], result[index + i]);
  }
  index += Stats::n_dim;

  for (auto i = 0; i < Types::n_dim; ++i) {
    copy(PKMN::Data::TYPE_CHAR_ARRAY[i], result[index + i]);
  }
  index += Types::n_dim;

  for (auto i = 0; i < Boosts::n_dim; ++i) {
    copy(Boosts::dim_labels()[i], result[index + i]);
  }
  index += Boosts::n_dim;

  for (auto i = 0; i < Volatiles::n_dim; ++i) {
    copy(Volatiles::dim_labels()[i], result[index + i]);
  }
  index += Volatiles::n_dim;

  for (auto i = 0; i < MoveSlots::n_dim; ++i) {
    copy(MoveSlots::dim_labels()[i], result[index + i]);
  }
  index += MoveSlots::n_dim;

  for (auto i = 0; i < Duration::n_dim; ++i) {
    copy(Duration::dim_labels()[i], result[index + i]);
  }
  index += Duration::n_dim;

  for (auto i = 0; i < Pokemon::n_dim; ++i) {
    copy(Pokemon::dim_labels[i], result[index + i]);
  }
  return result;
}

constexpr auto dim_labels = get_dim_labels();

} // namespace Active

}

} // namespace Encode