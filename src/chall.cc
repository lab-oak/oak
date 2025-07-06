#include <battle/init.h>
#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <data/strings.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <util/random.h>
#include <util/strings.h>

#include <csignal>
#include <iostream>
#include <vector>

using Strings::string_to_move;
using Strings::string_to_species;

using Obs = std::array<uint8_t, 16>;
using Node = Tree::Node<Exp3::JointBanditData<.03f, true>, Obs>;

bool run_search = true;

void handle_suspend(int signal) {
  std::cout << '!' << std::endl;
  run_search = false;
}

std::pair<pkmn_gen1_battle, pkmn_gen1_chance_durations>
parse_input(const std::string &line) {
  const auto side_strings = split(line, '|');
  if (side_strings.size() != 2) {
    throw std::runtime_error("Battle input string must have \'|\' ");
  }
  std::vector<std::vector<Init::Set>> sides{};
  for (const auto &side_string : side_strings) {
    std::vector<Init::Set> sets{};
    auto set_strings = split(side_string, ';');
    for (const auto &set_string : set_strings) {
      auto words = split(set_string, ' ');
      Init::Set set = parse_set(words);
      sets.push_back(set);
    }
    sides.push_back(sets);
  }
  prng device{std::random_device{}()};

  return Init::battle_data(sides[0], sides[1], device.uniform_64());
}

int main(int argc, char **argv) {

  std::signal(SIGTSTP, handle_suspend);

  prng device{std::random_device{}()};

  pkmn_gen1_battle_options options{};
  BattleData battle_data;
  while (true) {
    std::string line;
    std::cout << "Enter battle string: " << std::endl;
    std::getline(std::cin, line);
    try {
      const auto [battle, durations] = parse_input(line);
      battle_data = {battle, durations, {}};
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      continue;
    }
    break;
  }

  battle_data.result = Init::update(battle_data.battle, 0, 0, options);

  while (!pkmn_result_type(battle_data.result)) {
    std::cout << "\nBattle:" << std::endl;
    std::cout << Strings::battle_data_to_string(battle_data.battle,
                                                battle_data.durations, {});
    const auto [p1_choices, p2_choices] =
        Init::choices(battle_data.battle, battle_data.result);
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

    MonteCarlo::Model model{device.uniform_64()};
    MCTS search{};
    MCTS::Output output{};
    Node node{};

    while (true) {
      run_search = true;
      output = search.run(&run_search, node, battle_data, model, output);
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

    battle_data.result = Init::update(battle_data.battle, c1, c2, options);
    battle_data.durations =
        *pkmn_gen1_battle_options_chance_durations(&options);
  }

  return 0;
}
