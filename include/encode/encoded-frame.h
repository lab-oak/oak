#pragma once

#include <encode/battle.h>
#include <train/target.h>

namespace Encode {

struct EncodedFrame {
  uint8_t m, n;

  std::array<std::array<std::array<float, Pokemon::n_dim>, 5>, 2> pokemon;
  std::array<std::array<std::array<float, Active::n_dim>, 1>, 2> active;
  std::array<std::array<float, 6>, 2> hp;

  std::array<uint16_t, 9> p1_choice_indices;
  std::array<uint16_t, 9> p2_choice_indices;

  Train::Target target;
};

struct EncodedFrameInput {
  uint8_t *m;
  uint8_t *n;

  uint16_t *p1_choice_indices;
  uint16_t *p2_choice_indices;

  float *pokemon;
  float *active;
  float *hp;

  float *p1_empirical;
  float *p1_nash;
  float *p2_empirical;
  float *p2_nash;
  float *empirical_value;
  float *nash_value;
  float *score;

  void write(const EncodedFrame &frame) {
    *m++ = frame.m;
    *n++ = frame.n;

    std::memcpy(pokemon, reinterpret_cast<const float *>(&frame.pokemon),
                sizeof(float) * 2 * 5 * Pokemon::n_dim);
    pokemon += 2 * 5 * Pokemon::n_dim;
    std::memcpy(active, reinterpret_cast<const float *>(&frame.active),
                sizeof(float) * 2 * 1 * Active::n_dim);
    active += 2 * 1 * Active::n_dim;
    std::memcpy(hp, reinterpret_cast<const float *>(&frame.hp),
                sizeof(float) * 12);
    hp += 12;

    std::fill_n(p1_empirical, 9, 0.f);
    std::fill_n(p1_nash, 9, 0.f);
    std::fill_n(p1_choice_indices, 9, 0);
    std::fill_n(p2_empirical, 9, 0.f);
    std::fill_n(p2_nash, 9, 0.f);
    std::fill_n(p2_choice_indices, 9, 0);

    for (int i = 0; i < frame.m; ++i) {
      p1_empirical[i] = frame.target.p1_empirical[i];
      p1_nash[i] = frame.target.p1_nash[i];
      p1_choice_indices[i] = frame.p1_choice_indices[i];
    }
    for (int i = 0; i < frame.n; ++i) {
      p2_empirical[i] = frame.target.p2_empirical[i];
      p2_nash[i] = frame.target.p2_nash[i];
      p2_choice_indices[i] = frame.p2_choice_indices[i];
    }

    p1_empirical += 9;
    p1_nash += 9;
    p1_choice_indices += 9;

    p2_empirical += 9;
    p2_nash += 9;
    p2_choice_indices += 9;

    *empirical_value++ = frame.target.empirical_value;
    *nash_value++ = frame.target.nash_value;
    *score++ = frame.target.score;
  }
};

} // namespace Encode