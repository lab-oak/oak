#include <battle/init.h>
#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/util.h>

#include <data/moves.h>
#include <util/random.h>

#include <iostream>
#include <limits>
#include <type_traits>

#include <map>

using Key = std::pair<int, int>;
std::map<Key, size_t> map{};

std::array<std::array<Init::Set, 6>, 16> teams = SampleTeams::teams;
pkmn_gen1_battle old_battle;
pkmn_gen1_battle_options old_options;

const auto F = [&map, &old_battle, &old_options](auto b, auto options) {
  const auto &battle = View::ref(b);
  const pkmn_gen1_chance_durations &durations =
      *pkmn_gen1_battle_options_chance_durations(&options);

  for (auto s = 0; s < 2; ++s) {
    const auto &vol = battle.side(s).active().volatiles();
    const auto &duration = View::ref(durations).duration(s);

    if (vol.confusion()) {
      const auto set = static_cast<int>(vol.confusion_left());
      const auto seen = static_cast<int>(duration.confusion());
      std::pair<int, int> key{seen, set};
      map[key] += 1;

      if ((seen == 0) && (set > 0)) {
        const pkmn_gen1_chance_durations &old_durations =
            *pkmn_gen1_battle_options_chance_durations(&old_options);
        std::cout << "START" << std::endl;
        std::cout << Strings::battle_data_to_string(old_battle, old_durations,
                                                    0);
        std::cout << "________" << std::endl;
        std::cout << Strings::battle_data_to_string(b, durations, 0);
        std::cout << "END" << std::endl;
      }
    }
  }

  old_battle = b;
  old_options = options;
};

int main(int argc, char **argv) {

  for (auto &team : teams) {
    for (auto &set : team) {
      set.moves[0] = Data::Move::ConfuseRay;
    }
  }

  prng device{2323423344634};

  for (auto i = 0; i < std::atoll(argv[1]); ++i) {
    auto [battle, durations] = Init::battle_data(
        teams[i % 16], teams[(i + 1) % 16], device.uniform_64());
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