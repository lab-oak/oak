#pragma once

namespace PokeEngine {
constexpr float POKEMON_ALIVE = 30.0;
constexpr float POKEMON_HP = 100.0;

constexpr float POKEMON_ATTACK_BOOST = 30.0;
constexpr float POKEMON_DEFENSE_BOOST = 15.0;
constexpr float POKEMON_SPECIAL_ATTACK_BOOST = 30.0;
constexpr float POKEMON_SPECIAL_DEFENSE_BOOST = 15.0;
constexpr float POKEMON_SPEED_BOOST = 30.0;

constexpr float POKEMON_BOOST_MULTIPLIER_6 = 3.3;
constexpr float POKEMON_BOOST_MULTIPLIER_5 = 3.15;
constexpr float POKEMON_BOOST_MULTIPLIER_4 = 3.0;
constexpr float POKEMON_BOOST_MULTIPLIER_3 = 2.5;
constexpr float POKEMON_BOOST_MULTIPLIER_2 = 2.0;
constexpr float POKEMON_BOOST_MULTIPLIER_1 = 1.0;
constexpr float POKEMON_BOOST_MULTIPLIER_0 = 0.0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_1 = -1.0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_2 = -2.0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_3 = -2.5;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_4 = -3.0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_5 = -3.15;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_6 = -3.3;

constexpr float POKEMON_FROZEN = -40.0;
constexpr float POKEMON_ASLEEP = -25.0;
constexpr float POKEMON_PARALYZED = -25.0;
constexpr float POKEMON_TOXIC = -30.0;
constexpr float POKEMON_POISONED = -10.0;
constexpr float POKEMON_BURNED = -25.0;

constexpr float LEECH_SEED = -30.0;
constexpr float SUBSTITUTE = 40.0;
constexpr float CONFUSION = -20.0;

constexpr float REFLECT = 20.0;
constexpr float LIGHT_SCREEN = 20.0;
constexpr float STICKY_WEB = -25.0;
constexpr float AURORA_VEIL = 40.0;
constexpr float SAFE_GUARD = 5.0;
constexpr float TAILWIND = 7.0;

constexpr float STEALTH_ROCK = -10.0;
constexpr float SPIKES = -7.0;
constexpr float TOXIC_SPIKES = -7.0;

float eval_burned(const uint8_t *pokemon) {
  float multiplier = 0;
  for (int m = 0; m < 4; ++m) {
    auto moveid = pokemon[2 * m + Offsets::moves];
    const auto &move_data = Data::get_move_data(moveid);
    if (move_data.bp > 0 && Data::is_physical(move_data.type)) {
      ++multiplier;
    }
  }
  // don't make burn as punishing for special attackers
  return multiplier * POKEMON_BURNED;
}

float get_boost_multiplier(const uint8_t stage) {
  switch (stage) {
  case 0b00000000:
    return POKEMON_BOOST_MULTIPLIER_0;
  case 0b00001001:
    return POKEMON_BOOST_MULTIPLIER_NEG_1;
  case 0b00000001:
    return POKEMON_BOOST_MULTIPLIER_1;
  case 0b00001010:
    return POKEMON_BOOST_MULTIPLIER_NEG_2;
  case 0b00000010:
    return POKEMON_BOOST_MULTIPLIER_2;
  case 0b00001011:
    return POKEMON_BOOST_MULTIPLIER_NEG_3;
  case 0b10000011:
    return POKEMON_BOOST_MULTIPLIER_3;
  case 0b00001100:
    return POKEMON_BOOST_MULTIPLIER_NEG_4;
  case 0b00000100:
    return POKEMON_BOOST_MULTIPLIER_4;
  case 0b00001101:
    return POKEMON_BOOST_MULTIPLIER_NEG_5;
  case 0b00000101:
    return POKEMON_BOOST_MULTIPLIER_5;
  case 0b00001110:
    return POKEMON_BOOST_MULTIPLIER_NEG_6;
  case 0b00000110:
    return POKEMON_BOOST_MULTIPLIER_6;
  }
  return 0;
}

float evaluate_status(const uint8_t *pokemon) {
  const uint8_t byte = pokemon[Offsets::status];
  if (byte & 7) {
    return POKEMON_ASLEEP;
  }
  switch (byte) {
  case 0b00000000:
    return 0;
  case 0b00001000:
    return POKEMON_POISONED;
  case 0b00010000:
    return eval_burned(pokemon);
  case 0b00100000:
    return POKEMON_FROZEN;
  case 0b01000000:
    return POKEMON_PARALYZED;
  case 0b10001000:
    return POKEMON_TOXIC;
  }
  return 0;
}

float evaluate_pokemon(const uint8_t *data) {
  float score = POKEMON_ALIVE;
  const auto u16 = std::bit_cast<const uint16_t *>(data);
  score += POKEMON_HP * (float)u16[9] / u16[0];
  score += evaluate_status(data);
  return score;
}

float evaluate_side(const uint8_t *data) {
  float score;
  for (int p = 0; p < 6; ++p) {
    const uint8_t *pokemon = data + Offsets::pokemon * p;
    const auto u16 = std::bit_cast<const uint16_t *>(pokemon);
    if (u16[9]) {
      score += evaluate_pokemon(pokemon);
    }
  }

  // TODO bad casts
  const auto active = data + Offsets::active;
  score += get_boost_multiplier(active[12] & 15) * POKEMON_ATTACK_BOOST;
  score += get_boost_multiplier(active[12] >> 4) * POKEMON_DEFENSE_BOOST;
  score += get_boost_multiplier(active[13] & 15) * POKEMON_SPEED_BOOST;
  score += get_boost_multiplier(active[13] >> 4) * POKEMON_SPECIAL_ATTACK_BOOST;

  const auto volatiles = active + 16;
  const bool sub = volatiles[5] > 0;
  const bool light_screen = active[1] & 128;
  const bool reflect = active[2] & 1;

  score += sub * SUBSTITUTE;
  score += light_screen * LIGHT_SCREEN;
  score += reflect * REFLECT;
  return score;
}

float evaluate_battle(const pkmn_gen1_battle &battle) {
  float p1_score = evaluate_side(battle.bytes);
  float p2_score = evaluate_side(battle.bytes + Offsets::side);
  return sigmoid(p1_score - p2_score);
}

} // namespace PokeEngine
