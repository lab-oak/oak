#pragma once

#include <libpkmn/data/strings.h>
#include <libpkmn/pkmn.h>
#include <util/debug-log.h>
#include <util/random.h>
#include <util/strings.h>

namespace Util {

template <typename F, bool debug_log = true>
pkmn_result rollout_and_exec(auto &device, pkmn_gen1_battle &battle,
                             pkmn_gen1_battle_options &options, F func) {
  auto result = PKMN::update(battle, 0, 0, options);
  std::array<pkmn_choice, 9> choices;
  while (!pkmn_result_type(result)) {

    func(battle, options);

    auto seed = device.uniform_64();
    const auto m = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    const auto c1 = choices[seed % m];
    const auto n = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    seed >>= 32;
    const auto c2 = choices[seed % n];
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
  }
  return result;
}

} // namespace Util