#include <util/argparse.h>
#include <util/parse.h>
#include <util/policy.h>
#include <util/random.h>
#include <util/search.h>

#include <csignal>
#include <iostream>

struct ProgramArgs : public AgentArgs {
  std::optional<uint64_t> &seed = kwarg("seed", "Global program seed");
  bool &use_search_time =
      flag("--use-search-time",
           "Use --search-time value instead of ctrl+z to end search");
};

bool search_flag = true;

void handle_suspend(int signal) {
  std::cout << "Search Suspended." << std::endl;
  search_flag = false;
}

MCTS::BattleData parse_input(const std::string &line, uint64_t seed) {
  auto [battle, durations] = Parse::parse_battle(line, seed);
  MCTS::apply_durations(battle, durations);
  return {battle, durations, PKMN::result(battle)};
}

int main(int argc, char **argv) {

  std::signal(SIGTSTP, handle_suspend);

  auto args = argparse::parse<ProgramArgs>(argc, argv);

  auto agent = RuntimeSearch::Agent{
      .search_time = args.search_time,
      .bandit_name = args.bandit_name,
      .network_path = args.network_path,
      .discrete_network = args.use_discrete,
      .matrix_ucb_name = args.matrix_ucb_name,
      .flag = args.use_search_time ? nullptr : &search_flag};

  if (!args.seed.has_value()) {
    args.seed.emplace(std::random_device{}());
  }

  const auto policy_options =
      RuntimePolicy::Options{.mode = args.policy_mode,
                             .temp = args.policy_temp,
                             .min_prob = args.policy_min};

  mt19937 device{args.seed.value()};
  MCTS::BattleData battle_data;

  while (true) {
    std::string line;
    std::cout << "Input: battle-string" << std::endl;
    std::getline(std::cin, line);
    try {
      battle_data = parse_input(line, args.seed.value());
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      continue;
    }
    break;
  }

  // set the durations inside the options to start
  auto options = PKMN::options();
  pkmn_gen1_chance_options chance_options{};
  chance_options.durations = battle_data.durations;
  pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);

  while (!pkmn_result_type(battle_data.result)) {
    std::cout << "\nBattle:" << std::endl;
    std::cout << PKMN::battle_data_to_string(battle_data.battle,
                                             battle_data.durations);
    const auto [p1_choices, p2_choices] =
        PKMN::choices(battle_data.battle, battle_data.result);
    const auto [p1_labels, p2_labels] =
        PKMN::choice_labels(battle_data.battle, battle_data.result);

    std::cout << "\nP1 choices:" << std::endl;
    for (auto i = 0; i < p1_choices.size(); ++i) {
      std::cout << i << ": " << p1_labels[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "P2 choices:" << std::endl;
    for (auto i = 0; i < p2_choices.size(); ++i) {
      std::cout << i << ": " << p2_labels[i] << ' ';
    }
    std::cout << std::endl;

    std::cout << "Starting search. Suspend (Ctrl + Z) to stop." << std::endl;

    RuntimeSearch::Nodes nodes{};
    MCTS::Output output{};

    int p1_index = -1;
    int p2_index = -1;

    while (true) {
      search_flag = true;
      output = RuntimeSearch::run(device, battle_data, nodes, agent, output);
      print_output(output, battle_data.battle, p1_labels, p2_labels);
      std::cout << "Input: P1 index (P2 index); Negative index = sample."
                << std::endl;
      std::string line;
      if (!std::getline(std::cin, line)) {
        std::cerr << "Input stream error.\n";
        continue;
      }

      std::istringstream iss(line);
      if (iss >> p1_index) {
        if (p1_index < (int)output.m) {
          if (iss >> p2_index) {
            if (p2_index < (int)output.n) {
              break;
            }
          } else {
            break;
          }
        }
      }
      std::cout << "Invalid index (pair). Resuming search." << std::endl;
    }

    if (p1_index < 0) {
      std::cout << "Sampling Player 1" << std::endl;
      p1_index = RuntimePolicy::process_and_sample(
          device, output.p1_empirical, output.p1_nash, policy_options);
    }
    if (p2_index < 0) {
      std::cout << "Sampling Player 2" << std::endl;
      p2_index = RuntimePolicy::process_and_sample(
          device, output.p2_empirical, output.p2_nash, policy_options);
    }

    auto c1 = p1_choices[p1_index];
    auto c2 = p2_choices[p2_index];

    std::cout << "Actions: " << p1_labels[p1_index] << ' '
              << p2_labels[p2_index] << std::endl;
    sleep(1);

    battle_data.result = PKMN::update(battle_data.battle, c1, c2, options);
    battle_data.durations =
        *pkmn_gen1_battle_options_chance_durations(&options);
  }

  std::cout << "\nBattle:" << std::endl;
  std::cout << PKMN::battle_data_to_string(battle_data.battle,
                                           battle_data.durations);
  std::cout << "Score: " << PKMN::score(battle_data.result) << std::endl;

  return 0;
}
