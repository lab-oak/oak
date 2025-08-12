#include <data/teams.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/pkmn.h>
#include <libpkmn/strings.h>
#include <nn/network.h>
#include <search/bandit/exp3.h>
#include <search/mcts.h>
#include <util/parse.h>
#include <util/policy.h>
#include <util/random.h>
#include <util/search.h>

#include <csignal>
#include <iostream>
#include <vector>

using Strings::string_to_move;
using Strings::string_to_species;

using Obs = std::array<uint8_t, 16>;
using Node = Tree::Node<Exp3::JointBandit, Obs>;

bool run_search = true;

void handle_suspend(int signal) {
  std::cout << '!' << std::endl;
  run_search = false;
}

BattleData parse_input(const std::string &line) {
  mt19937 device{std::random_device{}()};
  const auto [battle, durations] =
      Parse::parse_battle(line, device.uniform_64());
  return {battle, durations, PKMN::result()};
}

int main(int argc, char **argv) {

  std::string default_bandit = "ucb-2.0";
  RuntimeSearch::Agent agent{"0", default_bandit, "mc", std::nullopt,
                             &run_search};
  if (argc > 1) {
    agent.network_path = argv[1];
  }
  if (argc > 2) {
    agent.bandit_name = argv[2];
  }

  std::cout << "network path: " << agent.network_path << std::endl;
  std::cout << "bandit algorithm: " << agent.bandit_name << std::endl;

  RuntimePolicy::Options policy_options{};

  std::signal(SIGTSTP, handle_suspend);

  mt19937 device{std::random_device{}()};

  auto options = PKMN::options();
  BattleData battle_data;

  while (true) {
    std::string line;
    std::cout << "Enter battle string: " << std::endl;
    std::getline(std::cin, line);
    try {
      battle_data = parse_input(line);
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      continue;
    }
    break;
  }

  while (!pkmn_result_type(battle_data.result)) {
    std::cout << "\nBattle:" << std::endl;
    std::cout << Strings::battle_data_to_string(battle_data.battle,
                                                battle_data.durations);
    const auto [p1_choices, p2_choices] =
        PKMN::choices(battle_data.battle, battle_data.result);
    const auto [p1_labels, p2_labels] =
        PKMN::choice_labels(battle_data.battle, battle_data.result);

    std::cout << "P1 choices:" << std::endl;
    for (auto i = 0; i < p1_choices.size(); ++i) {
      std::cout << i << ": " << p1_labels[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "P2 choices:" << std::endl;
    for (auto i = 0; i < p2_choices.size(); ++i) {
      std::cout << i << ": " << p2_labels[i] << ' ';
    }
    std::cout << std::endl;

    size_t p1_index, p2_index;

    std::cout << "Starting search. Suspend (Ctrl + Z) to stop." << std::endl;

    RuntimeSearch::Nodes nodes{};

    MCTS::Output output{};

    while (true) {
      run_search = true;
      output = RuntimeSearch::run(battle_data, nodes, agent, output);

      print_output(output, battle_data.battle, p1_labels, p2_labels);
      p2_index = RuntimePolicy::process_and_sample(
          device, output.p2_empirical, output.p2_nash, policy_options);
      std::cout << "(If the next input is a P1 choice index the battle is "
                   "advanced; otherwise search is resumed.):"
                << std::endl;
      std::string line;
      if (!std::getline(std::cin, line)) {
        std::cerr << "Input stream error.\n";
        continue;
      }

      std::istringstream iss(line);
      if (iss >> p1_index) {
        if (p1_index < output.m) {
          break;
        }
      }
      std::cout << "Invalid index. Resuming search." << std::endl;
    }

    auto c1 = p1_choices[p1_index];
    auto c2 = p2_choices[p2_index];

    std::cout << p1_labels[p1_index] << ' ' << p2_labels[p2_index] << std::endl;

    battle_data.result = PKMN::update(battle_data.battle, c1, c2, options);
    battle_data.durations =
        *pkmn_gen1_battle_options_chance_durations(&options);
  }

  return 0;
}
