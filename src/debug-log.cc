#include <util/random.h>

#include <teams/ou-sample-teams.h>
#include <util/debug-log.h>

#include <iostream>

int rollout_sample_teams_and_stream_debug_log(int argc, char **argv) {
  constexpr size_t log_size{128};
  using Teams::ou_sample_teams;

  if (argc != 4) {
    std::cout << "Usage: provide two sample team indices [0 - "
              << ou_sample_teams.size() << "] and a u64 seed.\n"
              << "Debug log is piped via stdout e.g.:\n"
              << "\t./build/debug-log 0 1 123456 | "
                 "./extern/engine/src/bin/pkmn-debug "
                 "> index.html"
              << std::endl;
    return 1;
  }

  auto p1 = std::atoll(argv[1]);
  auto p2 = std::atoll(argv[2]);
  uint64_t seed = std::atoll(argv[3]);

  if (p1 >= ou_sample_teams.size()) {
    std::cerr << "Invalid index for p1 team." << std::endl;
    return 1;
  }
  if (p2 >= ou_sample_teams.size()) {
    std::cerr << "Invalid index for p2 team." << std::endl;
    return 1;
  }

  auto battle = PKMN::battle(ou_sample_teams[p1], ou_sample_teams[p2], seed);
  pkmn_gen1_battle_options options{};
  std::array<pkmn_choice, 9> choices{};

  DebugLog<log_size> debug_log{};
  debug_log.set_header(battle);
  mt19937 device{seed};

  auto turns = 0;
  pkmn_choice c1{0};
  pkmn_choice c2{0};
  auto result = PKMN::result();
  while (!pkmn_result_type(result)) {
    const auto m = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    c1 = choices[device.random_int(m)];
    const auto n = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    c2 = choices[device.random_int(n)];

    result = debug_log.update(battle, c1, c2, options);
    ++turns;
  }

  for (const char c : debug_log.header) {
    std::cout << c;
  }
  for (const auto &frame : debug_log.frames) {
    for (const char c : frame) {
      std::cout << c;
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  return rollout_sample_teams_and_stream_debug_log(argc, argv);
}
