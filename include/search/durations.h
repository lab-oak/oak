#pragma once

#include <libpkmn/data.h>
#include <libpkmn/data/status.h>

void apply_durations(pkmn_gen1_battle &b, const pkmn_gen1_chance_durations &d) {

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

  auto &battle = View::ref(b);
  const auto &durations = View::ref(d);
  for (auto s = 0; s < 2; ++s) {
    auto &side = battle.sides[s];
    const auto &duration = durations.duration(s);

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

    if (const auto sleep = duration.sleep(0)) {
      auto &pokemon = side.stored();
      auto &status = reinterpret_cast<uint8_t &>(pokemon.status);

      if (Data::is_sleep(status) && Data::self(status)) {
        const uint8_t max = 8 - sleep;
        status &= 0b11111000; // keep rest bit, clear sleep remaining
        status |= static_cast<uint8_t>((battle.rng % max) + 1);
      }
    }
  }
}
