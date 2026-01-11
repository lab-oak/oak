#pragma once

#include <nn/battle/network.h>
#include <search/bandit/exp3.h>
#include <search/bandit/pexp3.h>
#include <search/bandit/pucb.h>
#include <search/bandit/ucb.h>
#include <search/bandit/ucb1.h>
#include <search/mcts.h>
#include <search/poke-engine-evalulate.h>
#include <util/strings.h>

#include <filesystem>
#include <fstream>
#include <optional>

namespace RuntimeSearch {

template <typename T> using UniqueNode = std::unique_ptr<MCTS::Node<T>>;
template <typename T> using UniqueTable = std::unique_ptr<MCTS::Table<T>>;

template <typename T> struct UniqueStats {
  UniqueNode<T> node;
  UniqueTable<T> table;
  void reset() noexcept {
    node.reset();
    table.reset();
  }
  void reset_stats() noexcept {
    if (node) {
      node->stats = {};
    }
    if (table) {
      table->entries[table->root_key] = {};
    }
  }
};

struct Nodes {
  UniqueStats<Exp3::JointBandit> exp3;
  UniqueStats<PExp3::JointBandit> pexp3;
  UniqueStats<UCB::JointBandit> ucb;
  UniqueStats<UCB1::JointBandit> ucb1;
  UniqueStats<PUCB::JointBandit> pucb;
  bool set;

  Nodes() { reset(); }

  auto &get(auto &unique_item) {
    if (set && !unique_item.get()) {
      throw std::runtime_error{"Nodes alread set elsewhere"};
    }
    unique_item =
        std::make_unique<std::remove_cvref_t<decltype(*unique_item)>>();
    set = true;
    return *unique_item;
  }

  void reset() {
    exp3.reset();
    pexp3.reset();
    ucb.reset();
    ucb1.reset();
    pucb.reset();
    set = false;
  }

  // For UCB game generation, if a node is kept and its stats are not reset
  // the empirical policies may have 0.0 at some actions.
  // This is a clumsy fix though. TODO
  void reset_stats() {
    // exp3.reset_stats();
    // pexp3.reset_stats();
    // ucb.reset_stats();
    // ucb1.reset_stats();
    // pucb.reset_stats();
  }

  bool update(auto i1, auto i2, const auto &obs) {
    reset();
    return false;
    // auto update_node = [&](auto &node) -> bool {
    // if (!node || !node->stats.is_init()) {
    //   return false;
    // }
    // auto child = node->children.find({i1, i2, obs});
    // if (child == node->children.end()) {
    //   node = std::make_unique<std::decay_t<decltype(*node)>>();
    //   return false;
    // } else {
    //   auto unique_child = std::make_unique<std::decay_t<decltype(*node)>>(
    //       std::move((*child).second));
    //   node.swap(unique_child);
    //   return true;
    // }
    // };
    // return update_node(exp3) || update_node(pexp3) || update_node(ucb) ||
    //        update_node(ucb1) || update_node(pucb);
  }
};

struct Agent {
  std::string search_time;
  std::string bandit_name;
  std::string network_path;
  bool discrete_network;
  std::string matrix_ucb_name;
  bool use_table;
  // valid if already loaded/cache set
  std::optional<NN::Battle::Network> network;
  bool *flag;

  bool uses_network() const {
    return !network_path.empty() && network_path != "mc" &&
           network_path != "montecarlo" && network_path != "monte-carlo" &&
           network_path != "fp";
  }

  void initialize_network(const pkmn_gen1_battle &b) {
    network.emplace();

    constexpr auto tries = 3;
    // sometimes reads fail because python is writing to that file. just retry
    for (auto i = 0; i < tries; ++i) {
      std::ifstream file{network_path};
      if (network.value().read_parameters(file)) {
        break;
      } else {
        if (i == (tries - 1)) {
          throw std::runtime_error{
              "Agent could not read network parameters at: " + network_path};
        }
        sleep(1);
      }
    }
    if (discrete_network) {
      network.value().enable_discrete();
    }
    network.value().fill_pokemon_caches(b);
  }

  std::string to_string() const {
    std::stringstream ss{};
    ss << "search_time: ";
    if (flag) {
      ss << "(flag)";
    } else {
      ss << search_time;
    }
    ss << " bandit_name: " << bandit_name;
    ss << " network_path: ";
    if (uses_network()) {
      ss << network_path;
    } else {
      ss << "(monte-carlo)";
    }
    return ss.str();
  }
};

auto run(auto &device, const MCTS::Input &input, Nodes &nodes, Agent &agent,
         MCTS::Output output = {}) {

  // Finally the MCTS::run call
  const auto run_5 = [&](auto dur, auto &model, auto &params,
                         auto &node) -> MCTS::Output {
    MCTS::Search search{};
    return search.run(device, dur, params, node, model, input, output);
  };

  // Whether to use MatrixUCB params or normal
  const auto run_4 = [&](auto dur, auto &model, auto &bandit_params,
                         auto &node) -> MCTS::Output {
    const auto &matrix_ucb_name = agent.matrix_ucb_name;
    if (!matrix_ucb_name.empty()) {
      const auto matrix_ucb_name_split =
          Parse::split(agent.matrix_ucb_name, '-');
      if (matrix_ucb_name_split.size() != 3) {
        throw std::runtime_error{"Could not parse MatrixUCB name: " +
                                 agent.matrix_ucb_name};
      }
      MCTS::MatrixUCBParams<std::remove_cvref_t<decltype(bandit_params)>>
          matrix_ucb_params{bandit_params};
      matrix_ucb_params.delay = std::stoull(matrix_ucb_name_split[0]);
      matrix_ucb_params.interval = std::stoull(matrix_ucb_name_split[1]);
      matrix_ucb_params.c = std::stof(matrix_ucb_name_split[2]);
      return run_5(dur, model, matrix_ucb_params, node);
    } else {
      return run_5(dur, model, bandit_params, node);
    }
  };

  // Whether to use tree nodes or transposition table
  const auto run_3 = [&](const auto dur, auto &model, const auto &params,
                         auto &both) {
    if (agent.use_table) {
      return run_4(dur, model, params, nodes.get(both.table));
    } else {
      return run_4(dur, model, params, nodes.get(both.node));
    }
  };

  // Parse bandit algorithm and parameters
  const auto run_2 = [&](auto dur, auto &model) {
    const auto bandit_name_split = Parse::split(agent.bandit_name, '-');

    if (bandit_name_split.size() < 2) {
      throw std::runtime_error("Could not parse bandit string: " +
                               agent.bandit_name);
    }

    const auto &name = bandit_name_split[0];
    const float f1 = std::stof(bandit_name_split[1]);

    if (name == "ucb") {
      UCB::Bandit::Params params{.c = f1};
      return run_3(dur, model, params, nodes.ucb);
    } else if (name == "ucb1") {
      UCB1::Bandit::Params params{.c = f1};
      return run_3(dur, model, params, nodes.ucb1);
    } else if (name == "pucb") {
      PUCB::Bandit::Params params{.c = f1};
      return run_3(dur, model, params, nodes.pucb);
    }

    float alpha = .05;
    if (bandit_name_split.size() >= 3) {
      alpha = std::stof(bandit_name_split[2]);
    }

    if (name == "exp3") {
      Exp3::Bandit::Params params{.gamma = f1,
                                  .one_minus_gamma = (1 - f1),
                                  .alpha = alpha,
                                  .one_minus_alpha = (1 - alpha)};
      return run_3(dur, model, params, nodes.exp3);
    } else if (name == "pexp3") {
      PExp3::Bandit::Params params{.gamma = f1,
                                   .one_minus_gamma = (1 - f1),
                                   .alpha = alpha,
                                   .one_minus_alpha = (1 - alpha)};
      return run_3(dur, model, params, nodes.pexp3);
    } else {
      throw std::runtime_error("Could not parse bandit string: " + name);
    }
  };

  // Whether to use a network, Monte Carlo, or Poke-Engine
  const auto search_1 = [&](const auto dur) {
    if (agent.uses_network()) {
      if (!agent.network.has_value()) {
        agent.initialize_network(input.battle);
      }
      return run_2(dur, agent.network.value());
    } else if (agent.network_path == "fp") {
      PokeEngine::Model model{};
      return run_2(dur, model);
    } else {
      MCTS::MonteCarlo model{};
      return run_2(dur, model);
    }
  };

  // Parse search budget
  const auto search_0 = [&]() {
    if (agent.flag != nullptr) {
      return search_1(agent.flag);
    }
    const auto pos = agent.search_time.find_first_not_of("0123456789");
    size_t number = std::stoll(agent.search_time.substr(0, pos));
    std::string unit =
        (pos == std::string::npos) ? "" : agent.search_time.substr(pos);
    if (unit.empty()) {
      return search_1(number);
    } else if (unit == "ms" || unit == "millisec" || unit == "milliseconds") {
      return search_1(std::chrono::milliseconds{number});
    } else if (unit == "s" || unit == "sec" || "seconds") {
      return search_1(std::chrono::seconds{number});
    } else {
      throw std::runtime_error("Invalid search duration specification: " +
                               agent.search_time);
    }
  };

  return search_0();
}

} // namespace RuntimeSearch