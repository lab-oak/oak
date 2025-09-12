#include <util/parse.h>
#include <util/random.h>
#include <util/search.h>

#include <exception>
#include <iostream>

std::string default_bandit = "exp3-0.03";
RuntimeSearch::Agent default_agent{"1_000_000", default_bandit, "mc",
                                   std::nullopt};

constexpr float small = .03;

struct Test {
  std::string position;
  float expected;
  float error = small;
  RuntimeSearch::Agent agent = default_agent;

  // Call after aggregate initialization
  void operator()() {
    auto battle_data = parse_input(position, std::random_device{}());
    auto options = PKMN::options();
    pkmn_gen1_chance_options chance_options{};
    chance_options.durations = battle_data.durations;
    pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);
    RuntimeSearch::Nodes nodes{};
    auto output = RuntimeSearch::run(battle_data, nodes, agent);
    bool success = std::abs(output.empirical_value - expected) <= error;
    if (!success) {
      std::cerr << position << std::endl;
      std::cerr << "value: " << output.empirical_value
                << " - expected: " << expected << std::endl;
      throw std::runtime_error{""};
    }
  }

private:
  static MCTS::BattleData parse_input(const std::string &line, uint64_t seed) {
    auto [battle, durations] = Parse::parse_battle(line, seed);
    return {battle, durations, PKMN::result(battle)};
  }
};

void confusion_duration() {
  Test{
      .position = "starmie seismictoss 1hp (conf:5) | snorlax bodyslam 1hp",
      .expected = 1.0,
      .error = 0,
  }();
  Test{
      .position = "starmie seismictoss 1hp (conf:4) | snorlax bodyslam 1hp",
      .expected = .5 + .5 / 2,
  }();
  Test{
      .position = "starmie seismictoss 1hp (conf:3) | snorlax bodyslam 1hp",
      .expected = .33 + .66 / 2,
  }();
  Test{
      .position = "starmie seismictoss 1hp (conf:2) | snorlax bodyslam 1hp",
      .expected = .25 + .75 / 2,
  }();
  Test{
      .position = "starmie seismictoss 1hp (conf:1) | snorlax bodyslam 1hp",
      .expected = .5,
  }();
}

void sleep() {
  Test{
      .position = "starmie seismictoss 1hp slp6 | snorlax seismictoss 1hp",
      .expected = 0.0,
      .error = 0,
  }();
  Test{
      .position = "starmie seismictoss 101hp slp0 | snorlax seismictoss 1hp",
      .expected = 1.0 / 7,
  }();
  Test{
      .position = "starmie seismictoss 101hp slp1 | snorlax seismictoss 1hp",
      .expected = 1.0 / 6,
  }();
  Test{
      .position = "starmie seismictoss 101hp slp2 | snorlax seismictoss 1hp",
      .expected = 1.0 / 5,
  }();
  Test{
      .position = "starmie seismictoss 101hp slp3 | snorlax seismictoss 1hp",
      .expected = 1.0 / 4,
  }();
  Test{
      .position = "starmie seismictoss 101hp slp4 | snorlax seismictoss 1hp",
      .expected = 1.0 / 3,
  }();
  Test{
      .position = "starmie seismictoss 101hp slp5 | snorlax seismictoss 1hp",
      .expected = 1.0 / 2,
  }();
  Test{
      .position = "starmie seismictoss 101hp slp6 | snorlax seismictoss 1hp",
      .expected = 1.0,
      .error = 0,
  }();
}

void run_tests() {
  confusion_duration();
  sleep();
}

int main() {
  run_tests();
  return 0;
}
