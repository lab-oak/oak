#include <util/parse.h>
#include <util/random.h>
#include <util/search.h>

#include <exception>
#include <iostream>

std::string default_bandit = "exp3-0.03";
RuntimeSearch::Agent default_agent{"1024", default_bandit, "mc", std::nullopt};

MCTS::BattleData parse_input(const std::string &line, uint64_t seed) {
  auto [battle, durations] = Parse::parse_battle(line, seed);
  MCTS::apply_durations(battle, durations);
  return {battle, durations, PKMN::result(battle)};
}

void assert_value(RuntimeSearch::Agent &agent, std::string str, float value,
                  float delta) {
  auto battle_data = parse_input(str, std::random_device{}());
  auto options = PKMN::options();
  pkmn_gen1_chance_options chance_options{};
  chance_options.durations = battle_data.durations;
  pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);
  RuntimeSearch::Nodes nodes{};
  MCTS::Output output{};
  output = RuntimeSearch::run(battle_data, nodes, agent, output);
  bool success = std::abs(output.empirical_value - value) <= delta;
  if (!success) {
    std::cerr << str << std::endl;
    std::cerr << "value: " << output.empirical_value << " - expected: " << value
              << std::endl;
    throw std::runtime_error{""};
  }
}

void run_tests() {
  assert_value(default_agent,
               "starmie seismictoss 1hp (conf:5) | snorlax bodyslam 1hp", 1.0,
               0);
}

int main() {
  run_tests();
  return 0;
}