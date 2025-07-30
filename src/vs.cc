#include <data/sample-teams.h>
#include <libpkmn/data.h>
#include <libpkmn/options.h>
#include <libpkmn/strings.h>
#include <nn/network.h>
#include <search/exp3.h>
#include <search/mcts.h>
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

using Obs = std::array<uint8_t, 16>;
using Exp3Node = Tree::Node<Exp3::JointBanditData<.03f, false>, Obs>;
using UCBNode = Tree::Node<UCB::JointBanditData<2.0f>, Obs>;

using Exp3_03 = Exp3::JointBanditData<0.3f>;
using Exp3_10 = Exp3::JointBanditData<0.1f>;
using Exp3_20 = Exp3::JointBanditData<0.2f>;
using UCB_05 = UCB::JointBanditData<0.5f>;
using UCB_10 = UCB::JointBanditData<1.0f>;
using UCB_20 = UCB::JointBanditData<2.0f>;

namespace RuntimeData {
bool terminated = false;
bool suspended = false;

std::atomic<size_t> score{};
std::atomic<size_t> n{};

} // namespace RuntimeData

struct SearchOptions {
  std::string battle_network_path = "mc";
  char count_mode = 'i';
  std::string bandit_name = "exp3-0.29";
  size_t count = {1 << 10};
};

struct PolicyOptions {
  double policy_temp = 1;
  double policy_min_prob = 0;
  char policy_mode = 'n';
  double mix_nash_weight = .5;
};

SearchOptions p1_search_options{};
SearchOptions p2_search_options{};

PolicyOptions p1_policy_options{};
PolicyOptions p2_policy_options{};

std::pair<int, int> sample(mt19937 &device, auto &p1_output, auto &p2_output) {

  const auto process_and_sample = [&](const auto &policy,
                                      const auto &policy_options) {
    const double t = policy_options.policy_temp;

    if (t <= 0) {
      throw std::runtime_error("Use positive policy power");
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
  };

  if (p1_policy_options.policy_mode == 'n') {
    return {process_and_sample(p1_output.p1_nash, p1_policy_options),
            process_and_sample(p2_output.p2_nash, p2_policy_options)};
  } else if (p1_policy_options.policy_mode == 'e') {
    return {process_and_sample(p1_output.p1_empirical, p1_policy_options),
            process_and_sample(p2_output.p2_empirical, p2_policy_options)};
  } else if (p1_policy_options.policy_mode == 'm') {
    const auto weighted_sum = [](const auto &a, const auto &b,
                                 const auto alpha) {
      auto result = a;
      std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                     [alpha](const auto &x, const auto &y) {
                       return alpha * x + (decltype(alpha))(1) - alpha * y;
                     });
      return result;
    };
    return {0, 0};
  }
  throw std::runtime_error("Invalid policy mode.");
}

void thread_fn(uint64_t seed) {
  mt19937 device{seed};

  const auto play = [&](const auto &p1_team, const auto &p2_team) {
    const auto battle = PKMN::battle(p1_team, p2_team, device.uniform_64());
    // const auto durations = PKMN::durations(p1_team, p2_team);
    const auto durations = PKMN::durations();
    BattleData battle_data{battle, durations};
    pkmn_gen1_battle_options battle_options{};
    battle_data.result = PKMN::update(battle_data.battle, 0, 0, battle_options);

    Train::CompressedFrames<> training_frames{battle_data.battle};

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

        if (p1_choices.size() > 1) {
          p1_output = RuntimeSearch::run<Exp3_03, Exp3_10, Exp3_20, UCB_05,
                                         UCB_10, UCB_20>(
              battle_data, p1_search_options.count,
              p1_search_options.count_mode, p1_search_options.bandit_name,
              p1_search_options.battle_network_path);
        } else {
          p1_output = RuntimeSearch::run<Exp3_03, Exp3_10, Exp3_20, UCB_05,
                                         UCB_10, UCB_20>(
              battle_data, 10, p1_search_options.count_mode,
              p1_search_options.bandit_name,
              p1_search_options.battle_network_path);
        }

        if (p2_choices.size() > 1) {
          p2_output = RuntimeSearch::run<Exp3_03, Exp3_10, Exp3_20, UCB_05,
                                         UCB_10, UCB_20>(
              battle_data, p2_search_options.count,
              p2_search_options.count_mode, p2_search_options.bandit_name,
              p2_search_options.battle_network_path);
        } else {
          p2_output = RuntimeSearch::run<Exp3_03, Exp3_10, Exp3_20, UCB_05,
                                         UCB_10, UCB_20>(
              battle_data, 10, p2_search_options.count_mode,
              p2_search_options.bandit_name,
              p2_search_options.battle_network_path);
        }

        const auto [p1_index, p2_index] = sample(device, p1_output, p2_output);

        const auto p1_choice = p1_choices[p1_index];
        const auto p2_choice = p2_choices[p2_index];

        battle_data.result = PKMN::update(battle_data.battle, p1_choice,
                                          p2_choice, battle_options);
        battle_data.durations = PKMN::durations(battle_options);
        ++updates;
      }

      const size_t score = PKMN::score2(battle_data.result);
      RuntimeData::score.fetch_add(score);
      RuntimeData::n.fetch_add(1);
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return;
    }
  };

  while (true) {
    const auto p1_team =
        SampleTeams::teams[device.random_int(SampleTeams::teams.size())];
    const auto p2_team =
        SampleTeams::teams[device.random_int(SampleTeams::teams.size())];

    play(p1_team, p2_team);
    play(p2_team, p1_team);

    if (RuntimeData::terminated) {
      return;
    }
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

  if (argc < 9) {
    std::cerr << "Side: count mode bandit-name net-path.\n Input (Side) p1, "
                 "(Side) p2."
              << std::endl;
    return 1;
  }

  p1_search_options.count = std::atoll(argv[1]);
  p1_search_options.count_mode = argv[2][0];
  p1_search_options.bandit_name = std::string{argv[3]};
  p1_search_options.battle_network_path = std::string(argv[4]);

  p2_search_options.count = std::atoll(argv[5]);
  p2_search_options.count_mode = argv[6][0];
  p2_search_options.bandit_name = std::string{argv[7]};
  p2_search_options.battle_network_path = std::string(argv[8]);

  size_t threads = 1;
  size_t seed = std::random_device{}();
  if (argc > 9) {
    threads = std::atoll(argv[9]);
  }
  if (argc > 10) {
    seed = std::atoll(argv[10]);
  }

  mt19937 device{seed};

  std::vector<std::thread> thread_pool{};
  for (auto t = 0; t < threads; ++t) {
    thread_pool.emplace_back(std::thread{&thread_fn, device.uniform_64()});
  }
  for (auto &thread : thread_pool) {
    thread.join();
  }

  std::cout << "score: "
            << (RuntimeData::score.load() / 2.0 / RuntimeData::n.load())
            << " over " << RuntimeData::n.load() << " games." << std::endl;
}