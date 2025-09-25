#include <libpkmn/data.h>
#include <libpkmn/strings.h>
#include <nn/battle/network.h>
#include <teams/ou-sample-teams.h>
#include <train/battle/compressed-frame.h>
#include <util/policy.h>
#include <util/random.h>
#include <util/search.h>
#include <util/team-building.h>

#include <atomic>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

auto inverse_sigmoid(const auto x) { return std::log(x) - std::log(1 - x); }

void print(const auto &data, const bool newline = true) {
  std::cout << data;
  if (newline) {
    std::cout << '\n';
  }
}

std::string container_string(const auto &v) {
  std::stringstream ss{};
  for (auto x : v) {
    if constexpr (std::is_same_v<decltype(x), std::string>) {
      ss << x << ' ';
    } else {
      ss << std::to_string(x) << ' ';
    }
  }
  return ss.str();
}

namespace RuntimeOptions {
size_t threads = 0;
double print_prob = .01;
double early_stop_log = 3.5;

RuntimeSearch::Agent p1_agent{"4096", "exp3-0.03", "mc"};
RuntimeSearch::Agent p2_agent{"4096", "exp3-0.03", "mc"};

RuntimePolicy::Options p1_policy_options{};
RuntimePolicy::Options p2_policy_options{};

namespace TeamGen {
// std::string teams_path = "";
std::string network_path = "";
NN::Build::Network network{};

double team_modify_prob = 0;
TeamBuilding::Omitter omitter{};
double battle_skip_prob = 0;
}; // namespace TeamGen

} // namespace RuntimeOptions

auto generate_team(mt19937 &device, const auto &base_team)
    -> Train::Build::Trajectory {
  using namespace RuntimeOptions::TeamGen;

  const auto base_team_vec =
      std::vector<PKMN::Set>(base_team.begin(), base_team.end());

  auto team = base_team_vec;
  const bool changed = omitter.shuffle_and_truncate(device, team);
  bool deleted = false;
  if (device.uniform() < team_modify_prob) {
    deleted = omitter.delete_info(device, team);
  }

  if (!deleted) {
    Train::Build::Trajectory trajectory{};
    trajectory.initial = trajectory.terminal = team;
    return trajectory;
  } else {
    const auto trajectory =
        TeamBuilding::rollout_build_network(device, network, team);
    assert(trajectory.updates.size() > 0);
    return trajectory;
  }
}

bool parse_options(int argc, char **argv) {
  using namespace RuntimeOptions;

  if (argc < 2) {
    std::cout << "Usage: ./vs [OPTIONS]\nArg '--threads=' is "
                 "required.\n--help for more."
              << std::endl;
    return true;
  }

  std::vector<char *> args(argv + 1, argv + argc);
  assert(args.size() == argc - 1);

  for (auto &a : args) {
    if (a == nullptr) {
      continue;
    }
    char *b = nullptr;
    std::swap(a, b);
    std::string arg{b};
    if (arg.starts_with("--help")) {
      std::cout << "TODO help text" << std::endl;
      return true;
    } else if (arg.starts_with("--threads=")) {
      threads = std::stoul(arg.substr(10));
    } else if (arg.starts_with("--p1-search-time=")) {
      p1_agent.search_time = arg.substr(17);
    } else if (arg.starts_with("--p2-search-time=")) {
      p2_agent.search_time = arg.substr(17);
    } else if (arg.starts_with("--p1-bandit-name=")) {
      p1_agent.bandit_name = arg.substr(17);
    } else if (arg.starts_with("--p2-bandit-name=")) {
      p2_agent.bandit_name = arg.substr(17);
    } else if (arg.starts_with("--p1-network-path=")) {
      p1_agent.network_path = arg.substr(18);
    } else if (arg.starts_with("--p2-network-path=")) {
      p2_agent.network_path = arg.substr(18);
      
    } else if (arg.starts_with("--max-pokemon=")) {
      TeamGen::omitter.max_pokemon = std::stoul(arg.substr(14));
    } else if (arg.starts_with("--build-network-path=")) {
      TeamGen::network_path = arg.substr(21);
    } else if (arg.starts_with("--team-modify-prob=")) {
      TeamGen::team_modify_prob = std::stod(arg.substr(19));
    } else if (arg.starts_with("--pokemon-delete-prob=")) {
      TeamGen::omitter.pokemon_delete_prob = std::stod(arg.substr(22));
    } else if (arg.starts_with("--move-delete-prob=")) {
      TeamGen::omitter.move_delete_prob = std::stod(arg.substr(19));
    } else {
      std::swap(a, b);
    }
  }

  for (auto a : args) {
    if (a != nullptr) {
      throw std::runtime_error{"Unrecognized arg: " + std::string(a)};
    }
  }

  return threads == 0;
}

namespace RuntimeData {
bool terminated = false;
bool suspended = false;

std::atomic<size_t> score{};
std::atomic<size_t> n{};
} // namespace RuntimeData

void thread_fn(uint64_t seed) {
  mt19937 device{seed};

  const auto play = [&](const auto &p1_team, const auto &p2_team) -> int {
    int score_2;

    auto battle = PKMN::battle(p1_team, p2_team, device.uniform_64());
    auto options = PKMN::options();
    const auto result = PKMN::update(battle, 0, 0, options);
    MCTS::BattleData battle_data{battle, PKMN::durations(), result};

    auto p1_agent_local = RuntimeOptions::p1_agent;
    auto p2_agent_local = RuntimeOptions::p2_agent;
    if (p1_agent_local.uses_network()) {
      p1_agent_local.read_network_parameters();
      p1_agent_local.network.value().fill_cache(battle);
    }
    if (p2_agent_local.uses_network()) {
      p2_agent_local.read_network_parameters();
      p2_agent_local.network.value().fill_cache(battle);
    }

    bool early_stop = false;
    size_t updates = 0;
    try {
      // playout game
      while (!pkmn_result_type(battle_data.result)) {

        while (RuntimeData::suspended) {
          sleep(1);
        }

        if (RuntimeData::terminated) {
          using Ignored = int;
          return Ignored{};
        }

        const auto [p1_choices, p2_choices] =
            PKMN::choices(battle_data.battle, battle_data.result);

        MCTS::Output p1_output, p2_output;
        int p1_index = 0;
        int p2_index = 0;
        int p1_early_stop = 0;
        int p2_early_stop = 0;
        if (p1_choices.size() > 1) {
          RuntimeSearch::Nodes nodes{};
          p1_output = RuntimeSearch::run(battle_data, nodes, p1_agent_local);
          p1_early_stop = inverse_sigmoid(p1_output.empirical_value) /
                          RuntimeOptions::early_stop_log;
          p1_index = process_and_sample(device, p1_output.p1_empirical,
                                        p1_output.p1_nash,
                                        RuntimeOptions::p1_policy_options);
        }
        if (p2_choices.size() > 1) {
          RuntimeSearch::Nodes nodes{};
          p2_output = RuntimeSearch::run(battle_data, nodes, p2_agent_local);
          p2_early_stop = inverse_sigmoid(p2_output.empirical_value) /
                          RuntimeOptions::early_stop_log;
          p2_index = process_and_sample(device, p2_output.p2_empirical,
                                        p2_output.p2_nash,
                                        RuntimeOptions::p2_policy_options);
        }

        // only if they have same sign and are both non zero
        if ((p1_early_stop * p2_early_stop) > 0) {
          early_stop = true;
          score_2 = (p1_early_stop > 0 ? 2 : 0);
          break;
        }

        const auto p1_choice = p1_choices[p1_index];
        const auto p2_choice = p2_choices[p2_index];

        if (device.uniform() < RuntimeOptions::print_prob) {
          print("GAME: " + std::to_string(RuntimeData::n.load()), false);
          print(" UPDATE: " + std::to_string(updates));
          print(PKMN::battle_data_to_string(battle_data.battle,
                                            battle_data.durations));
        }
        battle_data.result =
            PKMN::update(battle_data.battle, p1_choice, p2_choice, options);
        battle_data.durations = PKMN::durations(options);
        ++updates;
      }

      if (!early_stop) {
        score_2 = PKMN::score2(battle_data.result);
      }
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
    }
    return score_2;
  };

  while (true) {
    const auto p1_base_team = Teams::ou_sample_teams[device.random_int(
        Teams::ou_sample_teams.size())];
    const auto p2_base_team = Teams::ou_sample_teams[device.random_int(
        Teams::ou_sample_teams.size())];

    const auto p1_build_traj = generate_team(device, p1_base_team);
    const auto p2_build_traj = generate_team(device, p2_base_team);

    const auto p1_team = p1_build_traj.terminal;
    const auto p2_team = p2_build_traj.terminal;

    const auto s1 = play(p1_team, p2_team);
    const auto s2 = play(p2_team, p1_team);

    if (!RuntimeData::terminated) {
      RuntimeData::score.fetch_add(s1 + s2);
      RuntimeData::n.fetch_add(2);
    }

    if (RuntimeData::terminated) {
      return;
    }
  }

  return;
}

void progress_thread_fn(int sec) {
  while (true) {
    for (int s = 0; s < sec; ++s) {
      if (RuntimeData::terminated) {
        return;
      }
      sleep(1);
    }
    std::cout << "score: "
              << (RuntimeData::score.load() / 2.0 / RuntimeData::n.load())
              << " over " << RuntimeData::n.load() << " games." << std::endl;
  }
}

void handle_suspend(int signal) {
  RuntimeData::suspended = !RuntimeData::suspended;
  std::cout << (RuntimeData::suspended ? "Suspended." : "Resumed.")
            << std::endl;
}

void handle_terminate(int signal) {
  RuntimeData::terminated = true;
  RuntimeData::suspended = false;
  std::cout << "Terminated." << std::endl;
}

void setup(auto &device) {
  using namespace RuntimeOptions::TeamGen;
  if (!network_path.empty()) {
    std::ifstream file{RuntimeOptions::TeamGen::network_path};
    network.read_parameters(file);
  } else {
    network.initialize(device);
  }
}

int main(int argc, char **argv) {

  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  if (const bool exit_early = parse_options(argc, argv)) {
    return 1;
  }

  size_t seed = std::random_device{}();

  mt19937 device{seed};

  setup(device);

  std::vector<std::thread> thread_pool{};
  for (auto t = 0; t < RuntimeOptions::threads; ++t) {
    thread_pool.emplace_back(std::thread{&thread_fn, device.uniform_64()});
  }
  auto progress_thread = std::thread(&progress_thread_fn, 30);
  for (auto &thread : thread_pool) {
    thread.join();
  }
  progress_thread.join();

  std::cout << "score: "
            << (RuntimeData::score.load() / 2.0 / RuntimeData::n.load())
            << " over " << RuntimeData::n.load() << " games." << std::endl;
}