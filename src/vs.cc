#include <data/teams.h>
#include <libpkmn/data.h>
#include <libpkmn/strings.h>
#include <nn/network.h>
#include <search/exp3-policy.h>
#include <search/exp3.h>
#include <search/mcts.h>
#include <search/ucb-policy.h>
#include <search/ucb.h>
#include <train/compressed-frame.h>
#include <util/random.h>
#include <util/search.h>

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

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

namespace RuntimeData {
bool terminated = false;
bool suspended = false;

std::atomic<size_t> score{};
std::atomic<size_t> n{};
} // namespace RuntimeData

double print_prob = .01;
double early_stop_log = 3.5;

struct SearchOptions {
  std::string battle_network_path = "mc";
  char count_mode = 'i';
  std::string bandit_name = "exp3-0.029";
  size_t count = {1 << 10};

  std::string to_string() const {
    std::stringstream ss{};
    ss << "model: " << battle_network_path;
    ss << " mode: " << count_mode;
    ss << " bandit: " << bandit_name;
    ss << " count: " << count;
    return ss.str();
  }
};

struct PolicyOptions {
  double policy_temp = 1;
  double policy_min_prob = 0;
  char policy_mode = 'n';

  std::string to_string() const {
    std::stringstream ss{};
    ss << "temp: " << policy_temp;
    ss << " min-prob: " << policy_min_prob;
    ss << " mode: " << policy_mode;
    return ss.str();
  }
};

SearchOptions p1_search_options{};
SearchOptions p2_search_options{};

PolicyOptions p1_policy_options{};
PolicyOptions p2_policy_options{};

int process_and_sample(auto &device, const auto &empirical, const auto &nash,
                       const auto &policy_options) {
  const double t = policy_options.policy_temp;
  if (t <= 0) {
    throw std::runtime_error("Use positive policy power");
  }

  std::array<double, 9> policy{};
  if (policy_options.policy_mode == 'e') {
    policy = empirical;
  } else if (policy_options.policy_mode == 'n') {
    policy = nash;
  } else {
    throw std::runtime_error("bad policy mode.");
  }

  std::vector<double> p(policy.begin(), policy.end());
  double sum = 0;
  for (auto &val : p) {
    val = std::pow(val, t);
    sum += val;
  }
  if (policy_options.policy_min_prob > 0) {
    const double l = policy_options.policy_min_prob * sum;
    sum = 0;
    for (auto &val : p) {
      if (val < l)
        val = 0;
      sum += val;
    }
  }
  for (auto &val : p) {
    val /= sum;
  }

  const auto index = device.sample_pdf(p);
  return index;
}

void thread_fn(uint64_t seed) {
  mt19937 device{seed};

  const auto play = [&](const auto &p1_team, const auto &p2_team) {
    const auto battle = PKMN::battle(p1_team, p2_team, device.uniform_64());
    // RuntimeData::p1_network.fill_cache(battle);
    // RuntimeData::p2_network.fill_cache(battle);
    const auto durations = PKMN::durations();
    BattleData battle_data{battle, durations};
    auto battle_options = PKMN::options();
    battle_data.result = PKMN::update(battle_data.battle, 0, 0, battle_options);
    bool early_stop = false;

    // Train::CompressedFrames<> training_frames{battle_data.battle};

    size_t updates = 0;
    try {
      // playout game
      while (!pkmn_result_type(battle_data.result)) {

        while (RuntimeData::suspended) {
          sleep(1);
        }

        const auto [p1_choices, p2_choices] =
            PKMN::choices(battle_data.battle, battle_data.result);

        MCTS::Output p1_output, p2_output;
        int p1_index = 0;
        int p2_index = 0;
        int p1_early_stop = 0;
        int p2_early_stop = 0;
        if (p1_choices.size() > 1) {
          p1_output = RuntimeSearch::run(battle_data, p1_search_options.count,
                                         p1_search_options.count_mode,
                                         p1_search_options.bandit_name,
                                         p1_search_options.battle_network_path);
          p1_early_stop =
              inverse_sigmoid(p1_output.empirical_value) / early_stop_log;
          p1_index = process_and_sample(device, p1_output.p1_empirical,
                                        p1_output.p1_nash, p1_policy_options);
        }
        if (p2_choices.size() > 1) {
          p2_output = RuntimeSearch::run(battle_data, p2_search_options.count,
                                         p2_search_options.count_mode,
                                         p2_search_options.bandit_name,
                                         p2_search_options.battle_network_path);
          p2_early_stop =
              inverse_sigmoid(p2_output.empirical_value) / early_stop_log;
          p2_index = process_and_sample(device, p2_output.p2_empirical,
                                        p2_output.p2_nash, p2_policy_options);
        }

        // only if they have same sign and are both non zero
        if ((p1_early_stop * p2_early_stop) > 0) {
          early_stop = true;
          const size_t score = (p1_early_stop > 0 ? 2 : 0);
          RuntimeData::score.fetch_add(score);
          break;
        }

        const auto p1_choice = p1_choices[p1_index];
        const auto p2_choice = p2_choices[p2_index];

        if (device.uniform() < print_prob) {
          print("GAME: " + std::to_string(RuntimeData::n.load()), false);
          print(" UPDATE: " + std::to_string(updates));
          print(Strings::battle_data_to_string(battle_data.battle,
                                               battle_data.durations, {}),
                false);
          print("Values: " + std::to_string(p1_output.empirical_value) + " : " +
                std::to_string(p2_output.empirical_value));
          print("Times: " + std::to_string(p1_output.duration.count()) + " : " +
                std::to_string(p2_output.duration.count()));
          std::vector<std::string> p1_labels{};
          std::vector<std::string> p2_labels{};

          for (auto i = 0; i < p1_choices.size(); ++i) {
            p1_labels.push_back(Strings::side_choice_string(
                battle_data.battle.bytes, p1_choices[i]));
          }
          for (auto i = 0; i < p2_choices.size(); ++i) {
            p2_labels.push_back(Strings::side_choice_string(
                battle_data.battle.bytes + Layout::Sizes::Side, p2_choices[i]));
          }

          print("P1:");
          print(container_string(p1_labels));
          print("agent 1 empirical/nash");
          print(container_string(p1_output.p1_empirical));
          print(container_string(p1_output.p1_nash));
          print("agent 2 empirical/nash");
          print(container_string(p2_output.p1_empirical));
          print(container_string(p2_output.p1_nash));

          print("P2:");
          print(container_string(p2_labels));
          print("agent 1 empirical/nash");
          print(container_string(p1_output.p2_empirical));
          print(container_string(p1_output.p2_nash));
          print("agent 2 empirical/nash");
          print(container_string(p2_output.p2_empirical));
          print(container_string(p2_output.p2_nash));

          print(
              Strings::side_choice_string(battle_data.battle.bytes, p1_choice) +
              "/" +
              Strings::side_choice_string(battle_data.battle.bytes + 184,
                                          p2_choice));
        }
        battle_data.result = PKMN::update(battle_data.battle, p1_choice,
                                          p2_choice, battle_options);
        battle_data.durations = PKMN::durations(battle_options);
        ++updates;
      }

      if (!early_stop) {
        const size_t score = PKMN::score2(battle_data.result);
        RuntimeData::score.fetch_add(score);
      }
      RuntimeData::n.fetch_add(1);
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return;
    }
  };

  while (true) {
    const auto p1_team = Teams::teams[device.random_int(Teams::teams.size())];
    const auto p2_team = Teams::teams[device.random_int(Teams::teams.size())];

    play(p1_team, p2_team);
    play(p2_team, p1_team);

    if (RuntimeData::terminated) {
      return;
    }
  }
}

void progress_thread_fn(int sec) {
  while (!RuntimeData::terminated) {
    sleep(sec);
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

int main(int argc, char **argv) {

  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  if (argc < 11) {
    std::cerr << "Side: count mode bandit-name net-path policy-mode.\n Input "
                 "(Side) p1, "
                 "(Side) p2."
              << std::endl;
    return 1;
  }

  p1_search_options.count = std::atoll(argv[1]);
  p1_search_options.count_mode = argv[2][0];
  p1_search_options.bandit_name = std::string{argv[3]};
  p1_search_options.battle_network_path = std::string(argv[4]);
  p1_policy_options.policy_mode = argv[5][0];

  p2_search_options.count = std::atoll(argv[6]);
  p2_search_options.count_mode = argv[7][0];
  p2_search_options.bandit_name = std::string{argv[8]};
  p2_search_options.battle_network_path = std::string(argv[9]);
  p2_policy_options.policy_mode = argv[10][0];

  std::cout << "P1: " << p1_search_options.to_string() << ' '
            << p1_policy_options.to_string() << std::endl;
  std::cout << "P2: " << p2_search_options.to_string() << ' '
            << p2_policy_options.to_string() << std::endl;

  size_t threads = 1;
  size_t seed = std::random_device{}();
  if (argc > 11) {
    threads = std::atoll(argv[11]);
  }
  if (argc > 12) {
    print_prob = std::atof(argv[12]);
  }
  if (argc > 13) {
    seed = std::atoll(argv[13]);
  }

  std::cout << "threads: " << threads << std::endl;

  mt19937 device{seed};

  std::vector<std::thread> thread_pool{};
  for (auto t = 0; t < threads; ++t) {
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