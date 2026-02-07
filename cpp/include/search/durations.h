#pragma once

#include <libpkmn/data.h>

/*

For MCTS we need to acurately sample the turn possiblities for at the start of
each iteration

Just changing the RNG bytes will not work as expected, because showdown/libpkmn
use hidden variables for sleep, confusion

Accurate randomizing means randomly sampling those variables, the distributions
of which are dependent on public observations of how long the status condition
has been.

This means a state (in the markov sense) is not represented by the the bytes of
the battle But instead by the battle, excluding the hidden value bytes, and
including the durations bytes

*/

namespace MCTS {

void randomize_hidden_variables(pkmn_gen1_battle &b,
                                const pkmn_gen1_chance_durations &d) {

  static constexpr std::array<std::array<uint8_t, 40>, 4> multi{
      std::array<uint8_t, 40>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                              2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4},
      std::array<uint8_t, 40>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                              2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
      std::array<uint8_t, 40>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
      std::array<uint8_t, 40>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  auto &battle = PKMN::view(b);
  const auto &durations = PKMN::view(d);
  for (auto s = 0; s < 2; ++s) {
    auto &side = battle.sides[s];
    const auto &duration = durations.get(s);

    auto &vol = side.active.volatiles;

    if (const auto confusion = duration.confusion()) {
      const uint8_t max = 6 - (confusion + (confusion == 1));
      vol.set_confusion_left(
          static_cast<uint8_t>((battle.rng % max) + 1 + (confusion == 1)));
    }
    if (const auto disable = duration.disable()) {
      const uint8_t max = 9 - disable;
      vol.set_disable_left(static_cast<uint8_t>((battle.rng % max) + 1));
    }
    if (const auto attacking = duration.attacking()) {
      // bide and thrashing have same logic
      // just leave as separate if blocks for now
      if (vol.bide()) {
        if (attacking == 3) {
          vol.set_attacks(1);
        } else {
          vol.set_attacks(4 - (attacking + (battle.rng % 2)));
        }
      } else if (vol.thrashing()) {
        if (attacking == 3) {
          vol.set_attacks(1);
        } else {
          vol.set_attacks(4 - (attacking + (battle.rng % 2)));
        }
      } else {
        // in my testing this only happens when something is ko'd while binding
        assert(side.stored().hp == 0);
      }
    }
    if (const auto binding = duration.binding()) {
      const auto index = (battle.rng % 40);
      vol.set_attacks(multi[binding - 1][index]);
    }

    for (auto i = 0; i < 6; ++i) {
      if (const auto sleep = duration.sleep(i)) {
        auto &pokemon = side.get(i + 1);
        auto &status = reinterpret_cast<uint8_t &>(pokemon.status);

        if (PKMN::Data::is_sleep(status) && !PKMN::Data::self(status)) {
          const uint8_t max = 8 - sleep;
          status &= 0b11111000;
          status |= static_cast<uint8_t>((battle.rng % max) + 1);
        }
      }
    }
  }
}

consteval auto get_hidden_values_mask() {
  PKMN::Battle battle{};
  battle.rng = static_cast<uint64_t>(-1);
  for (auto &side : battle.sides) {
    auto &v = side.active.volatiles;
    v.set_confusion(7); // TODO max value for all 1s in those bits
    // remaining durations
    for (auto &pokemon : side.pokemon) {
      // we leave the top bit = rest intact
      pokemon.status = static_cast<PKMN::Data::Status>(0b111);
    }
  }
  constexpr auto n = PKMN::Layout::Sizes::Battle / 8;
  auto b = std::bit_cast<std::array<uint64_t, n>>(battle);
  for (auto &x : b) {
    x = !x;
  }
  return b;
}

constexpr auto hidden_values_mask = get_hidden_values_mask();

void clear_rng(pkmn_gen1_battle &battle) {
  auto &b =
      *reinterpret_cast<std::remove_cvref_t<decltype(hidden_values_mask)> *>(
          &battle);
  std::transform(b.begin(), b.end(), hidden_values_mask.begin(), b.begin(),
                 [](auto &x, const auto &y) { return x & y; });
}

} // namespace MCTS