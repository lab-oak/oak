#include <data/teams.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/pkmn.h>
#include <libpkmn/strings.h>
#include <util/parse.h>
#include <util/policy.h>
#include <util/random.h>
#include <util/search.h>

#include <csignal>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <map>

using Key = std::pair<int, int>;
std::map<Key, size_t> map{};

std::array<std::array<PKMN::Set, 6>, 16> teams = Teams::teams;
pkmn_gen1_battle old_battle;
pkmn_gen1_battle_options old_options;

const auto F = [&map, &old_battle, &old_options](auto b, auto options) {
  const auto &battle = View::ref(b);
  const pkmn_gen1_chance_durations &durations =
      *pkmn_gen1_battle_options_chance_durations(&options);

  for (auto s = 0; s < 2; ++s) {
    const auto &vol = battle.side(s).active().volatiles();
    const auto &duration = View::ref(durations).get(s);

    if (vol.disable_move() != 0) {
      // This is all you have to change besides setting the first move in main
      const auto set = static_cast<int>(vol.disable_left());
      const auto seen = static_cast<int>(duration.disable());
      std::pair<int, int> key{seen, set};
      map[key] += 1;

      if ((seen == 0) && (set > 0)) {
        const pkmn_gen1_chance_durations &old_durations =
            *pkmn_gen1_battle_options_chance_durations(&old_options);
        std::cout << "START" << std::endl;
        std::cout << Strings::battle_data_to_string(old_battle, old_durations);
        std::cout << "________" << std::endl;
        std::cout << Strings::battle_data_to_string(b, durations);
        std::cout << "END" << std::endl;
      }
    }
  }

  old_battle = b;
  old_options = options;
};

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Input: num-trials" << std::endl;
    return 1;
  }

  for (auto &team : teams) {
    for (auto &set : team) {
      set.moves[0] = Data::Move::Disable;
    }
  }

  mt19937 device{2323423344634};

  for (auto i = 0; i < std::atoll(argv[1]); ++i) {
    auto battle =
        PKMN::battle(teams[i % 16], teams[(i + 1) % 16], device.uniform_64());

    pkmn_gen1_battle_options options{};
    Util::rollout_and_exec(device, battle, options, F);
  }

  using Probs = std::array<float, 10>;

  std::map<int, Probs> total{};

  for (const auto [key, value] : map) {
    auto [seen, set] = key;
    auto &probs = total[seen];
    probs[set] += (float)value;
  }

  for (const auto [key, value] : total) {
    std::cout << key << std::endl;
    float sum = 0;
    for (auto i : value) {
      // std::cout << i << ' ';
      sum += i;
    }
    for (auto i : value) {
      std::cout << i / sum << ' ';
    }
    std::cout << std::endl;
  }

  return 0;
}