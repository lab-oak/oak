#include <data/teams.h>
#include <libpkmn/data.h>
#include <libpkmn/options.h>
#include <libpkmn/strings.h>
#include <nn/network.h>
#include <search/exp3.h>
#include <search/mcts.h>
#include <search/ucb.h>
#include <train/compressed-frames.h>
#include <util/random.h>

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
using UCBNode = Tree::Node<UCB::JointBanditData, Obs>;

namespace RuntimeData {
bool terminated = false;
bool suspended = false;

NN::Network p1_network{};
NN::Network p2_network{};

std::atomic<size_t> score{};
std::atomic<size_t> n{};

} // namespace RuntimeData

struct SearchData {
  NN::Network battle_network{};
  std::unique_ptr<Exp3Node> exp3_unique_node{std::make_unique<Exp3Node>()};
  std::unique_ptr<UCBNode> ucb_unique_node{std::make_unique<UCBNode>()};

  void clear() {
    exp3_unique_node = std::make_unique<Exp3Node>();
    ucb_unique_node = std::make_unique<UCBNode>();
  }
};

struct SearchOptions {
  std::string battle_network_path = "";
  char count_mode = 'i';
  char bandit_mode = 'e';
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

auto search(size_t c, mt19937 &device, const BattleData &battle_data,
            SearchData &search_data, const auto &options) {
  auto run_search_model = [&](auto &unique_node, auto &model,
                              const auto &options) {
    auto &node = *unique_node;

    assert(unique_node.get());

    MCTS search;
    if (options.count_mode == 'i' || options.count_mode == 'n') {
      return search.run(c, node, battle_data, model);
    } else if (options.count_mode == 't') {
      std::chrono::milliseconds ms{c};
      return search.run(ms, node, battle_data, model);
    }
    throw std::runtime_error("Invalid count mode char.");
  };

  auto run_search_node = [&](auto &unique_node, const auto &options) {
    if (options.battle_network_path.empty()) {
      MonteCarlo::Model model{device};
      return run_search_model(unique_node, model, options);
    } else {
      return run_search_model(unique_node, search_data.battle_network, options);
    }
  };

  if (options.bandit_mode == 'e') {
    return run_search_node(search_data.exp3_unique_node, options);
  } else if (options.bandit_mode == 'u') {
    return run_search_node(search_data.ucb_unique_node, options);
  }
  throw std::runtime_error("Invalid bandit mode char.");
}

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

    SearchData p1_search_data{RuntimeData::p1_network};
    SearchData p2_search_data{RuntimeData::p2_network};

    size_t updates = 0;
    try {
      // playout game
      while (!pkmn_result_type(battle_data.result)) {

        while (RuntimeData::suspended) {
          sleep(1);
        }

        p1_search_data.clear();
        p2_search_data.clear();

        const auto [p1_choices, p2_choices] =
            PKMN::choices(battle_data.battle, battle_data.result);

        MCTS::Output p1_output, p2_output;

        if (p1_choices.size() > 1) {
          p1_output = search(p1_search_options.count, device, battle_data,
                             p1_search_data, p1_search_options);
        } else {
          p1_output = search(10, device, battle_data, p1_search_data,
                             p1_search_options);
        }
        if (p2_choices.size() > 1) {
          p2_output = search(p2_search_options.count, device, battle_data,
                             p2_search_data, p2_search_options);
        } else {
          p2_output = search(10, device, battle_data, p2_search_data,
                             p2_search_options);
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
    const auto p1_team = Teams::teams[device.random_int(Teams::teams.size())];
    const auto p2_team = Teams::teams[device.random_int(Teams::teams.size())];

    play(p1_team, p2_team);
    play(p2_team, p1_team);

    if (RuntimeData::terminated) {
      return;
    }
  }
}

void prepare() {
  // create working dir
  std::string start_datetime =
      std::format("{:%F-%T}", std::chrono::floor<std::chrono::seconds>(
                                  std::chrono::system_clock::now()));

  const std::filesystem::path working_dir = start_datetime;
  std::error_code ec;
  const bool created = std::filesystem::create_directory(working_dir, ec);
  if (ec) {
    std::cerr << "Error creating directory: " << ec.message() << '\n';
  } else if (created) {
    std::cout << "Created directory " << start_datetime << std::endl;
  } else {
    throw std::runtime_error("Could not create datetime dir.");
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

  if (argc < 3) {
    std::cerr << "Input: net-path-1 net-path-2" << std::endl;
    return 1;
  }

  // read networks

  p1_search_options.battle_network_path = {argv[1]};
  p2_search_options.battle_network_path = {argv[2]};

  if (p1_search_options.battle_network_path != "mc") {
    std::ifstream file{p1_search_options.battle_network_path};
    if (!file) {
      std::cerr << "Cant open p1 network path" << std::endl;
      return 1;
    }
    const auto success = RuntimeData::p1_network.read_parameters(file);
    if (!success) {
      std::cerr << "Cant read p1 network data" << std::endl;
      return 1;
    }
  } else {
    p1_search_options.battle_network_path = "";
    std::cout << "p1 uses montecarlo" << std::endl;
  }

  if (p2_search_options.battle_network_path != "mc") {
    std::ifstream file{p2_search_options.battle_network_path};
    if (!file) {
      std::cerr << "Cant open p2 network path" << std::endl;
      return 1;
    }
    const auto success = RuntimeData::p2_network.read_parameters(file);
    if (!success) {
      std::cerr << "Cant read p2 network data" << std::endl;
      return 1;
    }
  } else {
    p2_search_options.battle_network_path = "";
    std::cout << "p2 uses montecarlo" << std::endl;
  }

  size_t threads = 1;
  size_t count = (1 << 10);
  size_t seed = std::random_device{}();
  if (argc >= 4) {
    threads = std::atoll(argv[3]);
  }
  if (argc >= 5) {
    count = std::atoll(argv[4]);
    p2_search_options.count = count;
    p1_search_options.count = count;
  }
  if (argc >= 6) {
    seed = std::atoll(argv[5]);
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