#include <data/teams.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/pkmn.h>
#include <libpkmn/strings.h>
#include <search/exp3.h>
#include <search/mcts.h>
#include <util/random.h>
#include <util/strings.h>

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
  return {battle, durations};
}

int main(int argc, char **argv) {

  std::signal(SIGTSTP, handle_suspend);

  mt19937 device{std::random_device{}()};

  pkmn_gen1_battle_options options{};
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

  battle_data.result = PKMN::update(battle_data.battle, 0, 0, options);

  while (!pkmn_result_type(battle_data.result)) {
    std::cout << "\nBattle:" << std::endl;
    std::cout << Strings::battle_data_to_string(battle_data.battle,
                                                battle_data.durations, {});
    const auto [p1_choices, p2_choices] =
        PKMN::choices(battle_data.battle, battle_data.result);
    std::vector<std::string> p1_labels{};
    std::vector<std::string> p2_labels{};
    for (auto i = 0; i < p1_choices.size(); ++i) {
      p1_labels.push_back(
          Strings::side_choice_string(battle_data.battle.bytes, p1_choices[i]));
    }
    for (auto i = 0; i < p2_choices.size(); ++i) {
      p2_labels.push_back(Strings::side_choice_string(
          battle_data.battle.bytes + Layout::Sizes::Side, p2_choices[i]));
    }

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

    size_t a, b;

    std::cout << "Starting search. Suspend (Ctrl + Z) to stop." << std::endl;

    Exp3::Bandit::Params bandit_params{.03f};
    MonteCarlo::Model model{device.uniform_64()};
    MCTS search{};
    MCTS::Output output{};
    Node node{};

    while (true) {
      run_search = true;
      output = search.run(&run_search, bandit_params, node, battle_data, model,
                          output);
      print_output(output, battle_data.battle, p1_labels, p2_labels);
      b = device.sample_pdf(output.p2_nash);
      std::cout << "(If the next input is a P1 choice index the battle is "
                   "advanced; otherwise search is resumed.):"
                << std::endl;
      std::string line;
      if (!std::getline(std::cin, line)) {
        std::cerr << "Input stream error.\n";
        continue;
      }

      std::istringstream iss(line);
      if (iss >> a) {
        if (a < output.m) {
          break;
        }
      }
      std::cout << "Invalid index. Resuming search." << std::endl;
    }

    auto c1 = p1_choices[a];
    auto c2 = p2_choices[b];

    std::cout << p1_labels[a] << ' ' << p2_labels[b] << std::endl;

    battle_data.result = PKMN::update(battle_data.battle, c1, c2, options);
    battle_data.durations =
        *pkmn_gen1_battle_options_chance_durations(&options);
  }

  return 0;
}
