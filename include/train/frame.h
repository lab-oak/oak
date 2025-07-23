#pragma once

namespace Train {

// Everything needed to train a value/policy net
struct Frame {
  uint8_t m, n;
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;
  std::array<float, 9> p1_empirical;
  std::array<float, 9> p1_nash;
  std::array<float, 9> p2_empirical;
  std::array<float, 9> p2_nash;
  float empirical_value;
  float nash_value;
  float score;
};

struct FrameInput {
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

  void write(const Frame &frame) {
    *m++ = frame.m;
    *n++ = frame.n;
    std::memcpy(battle, frame.battle.bytes, sizeof(pkmn_gen1_battle));
    battle += sizeof(pkmn_gen1_battle);
    std::memcpy(durations, frame.durations.bytes,
                sizeof(pkmn_gen1_chance_durations));
    durations += sizeof(pkmn_gen1_chance_durations);
    std::memcpy(result, &frame.result, sizeof(pkmn_result));
    result += sizeof(pkmn_result);
    std::memcpy(p1_choices, frame.p1_choices.data(), sizeof(pkmn_choice) * 9);
    p1_choices += sizeof(pkmn_choice) * 9;
    std::memcpy(p2_choices, frame.p2_choices.data(), sizeof(pkmn_choice) * 9);
    p2_choices += sizeof(pkmn_choice) * 9;
    std::fill_n(p1_empirical, 9, 0);
    std::fill_n(p1_nash, 9, 0);
    std::fill_n(p2_empirical, 9, 0);
    std::fill_n(p2_nash, 9, 0);
    for (int i = 0; i < frame.m; ++i) {
      p1_empirical[i] = frame.p1_empirical[i];
      p1_nash[i] = frame.p1_nash[i];
    }
    for (int i = 0; i < frame.n; ++i) {
      p2_empirical[i] = frame.p2_empirical[i];
      p2_nash[i] = frame.p2_nash[i];
    }
    p1_empirical += 9;
    p1_nash += 9;
    p2_empirical += 9;
    p2_nash += 9;
    *empirical_value++ = frame.empirical_value;
    *nash_value++ = frame.nash_value;
    *score++ = frame.score;
  }
};

} // namespace Train