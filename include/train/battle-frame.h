#pragma once

#include <train/target.h>

namespace Train {

// Everything needed to train a value/policy net
struct BattleFrame {
  uint8_t m, n;
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;
  Target target;
};

struct BattleFrameInput {
  uint8_t *m;
  uint8_t *n;
  uint8_t *battle;
  uint8_t *durations;
  uint8_t *result;
  uint8_t *p1_choices;
  uint8_t *p2_choices;
  float *p1_empirical;
  float *p1_nash;
  float *p2_empirical;
  float *p2_nash;
  float *empirical_value;
  float *nash_value;
  float *score;

  void write(const BattleFrame &frame) {
    *m++ = frame.m;
    *n++ = frame.n;
    std::memcpy(battle, frame.battle.bytes, sizeof(pkmn_gen1_battle));
    battle += sizeof(pkmn_gen1_battle);
    std::memcpy(durations, frame.durations.bytes,
                sizeof(pkmn_gen1_chance_durations));
    durations += sizeof(pkmn_gen1_chance_durations);
    std::memcpy(result, &frame.result, sizeof(pkmn_result));
    result += sizeof(pkmn_result);
    std::fill_n(p1_empirical, 9, 0);
    std::fill_n(p1_nash, 9, 0);
    std::fill_n(p2_empirical, 9, 0);
    std::fill_n(p2_nash, 9, 0);
    for (int i = 0; i < frame.m; ++i) {
      p1_empirical[i] = frame.target.p1_empirical[i];
      p1_nash[i] = frame.target.p1_nash[i];
      p1_choices[i] = frame.p1_choices[i];
    }
    for (int i = 0; i < frame.n; ++i) {
      p2_empirical[i] = frame.target.p2_empirical[i];
      p2_nash[i] = frame.target.p2_nash[i];
      p2_choices[i] = frame.p2_choices[i];
    }
    p1_empirical += 9;
    p1_nash += 9;
    p2_empirical += 9;
    p2_nash += 9;
    p1_choices += 9;
    p2_choices += 9;
    *empirical_value++ = frame.target.empirical_value;
    *nash_value++ = frame.target.nash_value;
    *score++ = frame.target.score;
  }
};

} // namespace Train