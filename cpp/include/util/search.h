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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <thread>

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
  void reset_node_stats() noexcept {
    if (node) {
      node->stats = {};
    }
  }

  void reset_table_stats(const auto &state) {
    if (table) {
      table->entries[state.s1.last ^ state.s2.last] = {};
    }
  }
};
// TODO just let claude unfuck this
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
  void reset_node_stats() {
    exp3.reset_node_stats();
    pexp3.reset_node_stats();
    ucb.reset_node_stats();
    ucb1.reset_node_stats();
    pucb.reset_node_stats();
  }

  void reset_table_stats(const auto &state) {
    exp3.reset_table_stats(state);
    pexp3.reset_table_stats(state);
    ucb.reset_table_stats(state);
    ucb1.reset_table_stats(state);
    pucb.reset_table_stats(state);
  }

  bool update(auto i1, auto i2, const auto &obs) {
    auto update_node = [&](auto &node) -> bool {
      if (!node || !node->stats.is_init()) {
        return false;
      }
      auto child = node->children.find({i1, i2, obs});
      if (child == node->children.end()) {
        node = std::make_unique<std::decay_t<decltype(*node)>>();
        return false;
      } else {
        auto unique_child = std::make_unique<std::decay_t<decltype(*node)>>(
            std::move((*child).second));
        node.swap(unique_child);
        return true;
      }
    };
    // No action is required to update a table
    return update_node(exp3.node) || update_node(pexp3.node) ||
           update_node(ucb.node) || update_node(ucb1.node) ||
           update_node(pucb.node);
  }
};

struct AgentParams {
  std::string search_budget;
  std::string bandit;
  std::string eval;
  std::string matrix_ucb;
  bool discrete;
  bool table;
};

struct Agent : AgentParams {

  std::unique_ptr<NN::Battle::NetworkBase> pointer;

  Agent() = default;
  Agent(const Agent &other) : AgentParams{static_cast<AgentParams>(other)} {}

  constexpr bool operator==(const Agent &other) const {
    return static_cast<AgentParams>(*this) == static_cast<AgentParams>(other);
  }

  bool is_monte_carlo() const {
    return eval.empty() || eval == "mc" || eval == "montecarlo" ||
           eval == "monte-carlo";
  }
  bool is_foul_play() const { return eval == "fp"; }
  bool uses_network() const { return !is_monte_carlo() && !is_fp(); }

  void initialize_network(const pkmn_gen1_battle &b) {
    network = std::make_unique<NN::Battle::Network<NN::Battle::MainNet>>();
    // sometimes reads fail because python is writing to that file. just retry
    const auto try_read_parameters = [&network, eval]() {
      constexpr auto tries = 3;
      bool read_success = false;
      for (auto i = 0; i < tries; ++i) {
        std::ifstream file{eval};
        if (network->read_parameters(file)) {
          read_success = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      if (!read_success) {
        throw std::runtime_error{
            "Agent could not read network parameters at: " + eval};
      }
    };
    try_read_parameters();
    if (discrete) {
      network = try_make_quantized_network(network);
    }
    network->init_caches(b);
  }
};

auto run(auto &device, const MCTS::Input &input, Nodes &nodes, Agent &agent,
         MCTS::Output output = {}, bool *const flag = nullptr) {

  const auto search = [&](auto dur, auto &model, auto &params,
                          auto &node) -> MCTS::Output {
    MCTS::Search s{};
    const auto o = s.run(device, dur, params, node, model, input, output);
    // std::cout << "depth: " << (float)s.total_depth / o.iterations
    //           << " errors: " << s.errors << std::endl;
    return o;
  };

  const auto parse_matrix_ucb_and_search = [&](auto dur, auto &model,
                                               auto &bandit_params,
                                               auto &node) -> MCTS::Output {
    const auto &matrix_ucb = agent.matrix_ucb;
    if (!matrix_ucb.empty()) {
      const auto matrix_ucb_split = Parse::split(agent.matrix_ucb, '-');
      if (matrix_ucb_split.size() != 4) {
        throw std::runtime_error{"Could not parse MatrixUCB name: " +
                                 agent.matrix_ucb};
      }
      MCTS::MatrixUCBParams<std::remove_cvref_t<decltype(bandit_params)>>
          matrix_ucb_params{bandit_params};
      matrix_ucb_params.delay = std::stoull(matrix_ucb_split[0]);
      matrix_ucb_params.interval = std::stoull(matrix_ucb_split[1]);
      matrix_ucb_params.minimum = std::stoull(matrix_ucb_split[2]);
      matrix_ucb_params.c = std::stof(matrix_ucb_split[3]);
      return search(dur, model, matrix_ucb_params, node);
    } else {
      return search(dur, model, bandit_params, node);
    }
  };

  const auto parse_heap_and_search = [&](const auto dur, auto &model,
                                         const auto &params, auto &both) {
    if (agent.use_table) {
      auto &table = nodes.get(both.table);
      table.hasher = {device};
      return parse_matrix_ucb_and_search(dur, model, params, table);
    } else {
      return parse_matrix_ucb_and_search(dur, model, params,
                                         nodes.get(both.node));
    }
  };

  const auto parse_bandit_and_search = [&](auto dur, auto &model) {
    const auto bandit_split = Parse::split(agent.bandit, '-');

    if (bandit_split.size() < 2) {
      throw std::runtime_error("Could not parse bandit string: " +
                               agent.bandit);
    }

    const auto &name = bandit_split[0];
    const float f1 = std::stof(bandit_split[1]);

    if (name == "ucb") {
      UCB::Bandit::Params params{.c = f1};
      return parse_heap_and_search(dur, model, params, nodes.ucb);
    } else if (name == "ucb1") {
      UCB1::Bandit::Params params{.c = f1};
      return parse_heap_and_search(dur, model, params, nodes.ucb1);
    } else if (name == "pucb") {
      PUCB::Bandit::Params params{.c = f1};
      return parse_heap_and_search(dur, model, params, nodes.pucb);
    }

    float alpha = .05;
    if (bandit_split.size() >= 3) {
      alpha = std::stof(bandit_split[2]);
    }

    if (name == "exp3") {
      Exp3::Bandit::Params params{.gamma = f1,
                                  .one_minus_gamma = (1 - f1),
                                  .alpha = alpha,
                                  .one_minus_alpha = (1 - alpha)};
      return parse_heap_and_search(dur, model, params, nodes.exp3);
    } else if (name == "pexp3") {
      PExp3::Bandit::Params params{.gamma = f1,
                                   .one_minus_gamma = (1 - f1),
                                   .alpha = alpha,
                                   .one_minus_alpha = (1 - alpha)};
      return parse_heap_and_search(dur, model, params, nodes.pexp3);
    } else {
      throw std::runtime_error("Could not parse bandit string: " + name);
    }
  };

  const auto parse_eval_and_search = [&](const auto dur) {
    if (agent.is_monte_carlo()) {
      MCTS::MonteCarlo model{};
      return parse_bandit_and_search(dur, model);
    } else if (agent.is_foul_play()) {
      PokeEngine::Model model{};
      return parse_bandit_and_search(dur, model);
    } else {
      if (!agent.network) {
        agent.initialize_network(input.battle);
      } else {
        if (!agent.network->check_cache()) {
          throw std::runtime_error{"Cache mismatch."};
        }
      }
      return parse_bandit_and_search(dur, agent.network.value());
    }
  };

  const auto parse_budget_and_search = [&]() {
    if (flag != nullptr) {
      return parse_eval_and_search(flag);
    }
    const auto pos = agent.search_budget.find_first_not_of("0123456789");
    size_t number = std::stoll(agent.search_budget.substr(0, pos));
    std::string unit =
        (pos == std::string::npos) ? "" : agent.search_budget.substr(pos);
    if (unit.empty()) {
      return parse_eval_and_search(number);
    } else if (unit == "ms" || unit == "millisec" || unit == "milliseconds") {
      return parse_eval_and_search(std::chrono::milliseconds{number});
    } else if (unit == "s" || unit == "sec" || unit == "seconds") {
      return parse_eval_and_search(std::chrono::seconds{number});
    } else {
      throw std::runtime_error("Invalid search duration specification: " +
                               agent.search_budget);
    }
  };

  return parse_budget_and_search();
}

} // namespace RuntimeSearch