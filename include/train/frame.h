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
  // std::array<float, 9> p1_policy_mask;
  // std::array<float, 9> p1_policy_mask;
  std::array<float, 9> p1_empirical;
  std::array<float, 9> p1_nash;
  std::array<float, 9> p2_empirical;
  std::array<float, 9> p2_nash;
  float eval;
  float score;
};

} // namespace Train