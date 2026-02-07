#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <train/battle/frame.h>
#include <train/battle/target.h>

namespace Encode {

namespace Battle {

struct Frame {
  uint8_t m, n;

  using PokemonEmbedding = std::array<float, Pokemon::n_dim>;
  using ActiveEmbedding = std::array<float, Active::n_dim>;

  std::array<std::array<PokemonEmbedding, 5>, 2> pokemon;
  std::array<std::array<ActiveEmbedding, 1>, 2> active;
  std::array<std::array<float, 6>, 2> hp;

  std::array<int64_t, 9> p1_choice_indices;
  std::array<int64_t, 9> p2_choice_indices;

  Frame() = default;
  Frame(const Train::Battle::Frame &frame)
      : pokemon{}, active{}, hp{}, p1_choice_indices{}, p2_choice_indices{} {

    const auto &battle = PKMN::view(frame.battle);
    const auto &durations = PKMN::view(frame.durations);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      if (stored.hp == 0) {
        hp[s][0] = 0;
        std::fill(active[s][0].begin(), active[s][0].end(), 0);
      } else {
        hp[s][0] = (float)stored.hp / stored.stats.hp;
        Encode::Battle::Active::write(stored, side.active, duration,
                                      active[s][0].data());
      }

      for (auto slot = 2; slot <= 6; ++slot) {
        const auto id = side.order[slot - 1];
        if (id == 0) {
          hp[s][slot - 1] = 0;
          std::fill(pokemon[s][slot - 2].begin(), pokemon[s][slot - 2].end(),
                    0);
        } else {
          const auto &poke = side.pokemon[id - 1];
          if (poke.hp == 0) {
            hp[s][slot - 1] = 0;
            std::fill(pokemon[s][slot - 2].begin(), pokemon[s][slot - 2].end(),
                      0);
          } else {
            const auto sleep = duration.sleep(slot - 1);
            hp[s][slot - 1] = (float)poke.hp / poke.stats.hp;
            Encode::Battle::Pokemon::write(poke, sleep,
                                           pokemon[s][slot - 2].data());
          }
        }
      }
    }

    m = frame.m;
    n = frame.n;
    for (auto i = 0; i < frame.m; ++i) {
      p1_choice_indices[i] = Encode::Battle::Policy::get_index(
          battle.sides[0], frame.p1_choices[i]);
    }
    for (auto i = 0; i < frame.n; ++i) {
      p2_choice_indices[i] = Encode::Battle::Policy::get_index(
          battle.sides[1], frame.p2_choices[i]);
    }
  }
};

struct FrameInput {
  uint8_t *m;
  uint8_t *n;

  int64_t *p1_choice_indices;
  int64_t *p2_choice_indices;

  float *pokemon;
  float *active;
  float *hp;

  uint32_t *iterations;
  float *p1_empirical;
  float *p1_nash;
  float *p2_empirical;
  float *p2_nash;
  float *empirical_value;
  float *nash_value;
  float *score;

  FrameInput index(auto i) const {
    auto copy = *this;
    copy.m += i;
    copy.n += i;
    copy.pokemon += i * 2 * 5 * Pokemon::n_dim;
    copy.active += i * 2 * 1 * Active::n_dim;
    copy.hp += i * 12;
    copy.p1_empirical += i * 9;
    copy.p1_nash += i * 9;
    copy.p1_choice_indices += i * 9;
    copy.iterations += i;
    copy.p2_empirical += i * 9;
    copy.p2_nash += i * 9;
    copy.p2_choice_indices += i * 9;
    copy.empirical_value += i;
    copy.nash_value += i;
    copy.score += i;
    return copy;
  }

  void write(const Frame &frame, const Train::Battle::Target &target) {
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

    *iterations++ = target.iterations;
    std::fill_n(p1_empirical, 9, 0.f);
    std::fill_n(p1_nash, 9, 0.f);
    // we use 'one after last' index to encode invalid index.
    // this way we can just cat a -neg inf onto the logits for softmax
    std::fill_n(p1_choice_indices, 9, Encode::Battle::Policy::n_dim);
    std::fill_n(p2_empirical, 9, 0.f);
    std::fill_n(p2_nash, 9, 0.f);
    std::fill_n(p2_choice_indices, 9, Encode::Battle::Policy::n_dim);

    for (int i = 0; i < frame.m; ++i) {
      p1_empirical[i] = target.p1_empirical[i];
      p1_nash[i] = target.p1_nash[i];
      p1_choice_indices[i] = frame.p1_choice_indices[i];
    }
    for (int i = 0; i < frame.n; ++i) {
      p2_empirical[i] = target.p2_empirical[i];
      p2_nash[i] = target.p2_nash[i];
      p2_choice_indices[i] = frame.p2_choice_indices[i];
    }

    p1_empirical += 9;
    p1_nash += 9;
    p1_choice_indices += 9;

    p2_empirical += 9;
    p2_nash += 9;
    p2_choice_indices += 9;

    *empirical_value++ = target.empirical_value;
    *nash_value++ = target.nash_value;
    *score++ = target.score;
  }
};

} // namespace Battle

} // namespace Encode