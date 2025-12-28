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

template <typename T>
using UniqueNode = std::unique_ptr<Tree::Node<T, MCTS::Obs>>;

struct Nodes {
  UniqueNode<Exp3::JointBandit> exp3;
  UniqueNode<PExp3::JointBandit> pexp3;
  UniqueNode<UCB::JointBandit> ucb;
  UniqueNode<UCB1::JointBandit> ucb1;
  UniqueNode<PUCB::JointBandit> pucb;
  bool set;

  Nodes() { reset(); }

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
    if (exp3) {
      exp3->stats() = {};
    }
    if (pexp3) {
      pexp3->stats() = {};
    }
    if (ucb) {
      ucb->stats() = {};
    }
    if (ucb1) {
      ucb1->stats() = {};
    }
    if (pucb) {
      pucb->stats() = {};
    }
  }

  bool update(auto i1, auto i2, const auto &obs) {
    auto update_node = [&](auto &node) -> bool {
      if (!node || !node->is_init()) {
        return false;
      }
      auto child = node->find(i1, i2, obs);
      if (child == node->_map.end()) {
        node = std::make_unique<std::decay_t<decltype(*node)>>();
        return false;
      } else {
        auto unique_child = std::make_unique<std::decay_t<decltype(*node)>>(
            std::move((*child).second));
        node.swap(unique_child);
        return true;
      }
    };
    return update_node(exp3) || update_node(pexp3) || update_node(ucb) ||
           update_node(ucb1) || update_node(pucb);
  }

  size_t count() const {
    if (exp3) {
      return Tree::count(*exp3);
    }
    if (pexp3) {
      return Tree::count(*pexp3);
    }
    if (ucb) {
      return Tree::count(*ucb);
    }
    if (ucb1) {
      return Tree::count(*ucb1);
    }
    if (pucb) {
      return Tree::count(*pucb);
    }
    return 0;
  }
};

struct Agent {
  std::string search_time;
  std::string bandit_name;
  std::string network_path;
  bool discrete_network;
  std::string matrix_ucb_name;
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

auto run(auto &input, Nodes &nodes, Agent &agent, MCTS::Output output = {}) {

  MCTS::BattleData battle_data{};
  using input_t = std::remove_cvref_t<decltype(input)>;
  if constexpr (std::is_same_v<input_t, pkmn_gen1_battle>) {
    battle_data.battle = input;
    battle_data.result = PKMN::result(input);
  } else if constexpr (std::is_same_v<input_t,
                                      std::pair<pkmn_gen1_battle,
                                                pkmn_gen1_chance_durations>>) {
    battle_data.battle = input.first;
    battle_data.durations = input.second;
    battle_data.result = PKMN::result(input);
  } else if constexpr (std::is_same_v<input_t, MCTS::BattleData>) {
    battle_data = input;
  } else {
    assert(false);
  }

  const auto run_3 = [&](auto dur, auto &model, auto &params,
                         auto &node) -> MCTS::Output {
    MCTS::Search search{};
    const auto &matrix_ucb_name = agent.matrix_ucb_name;
    if (!matrix_ucb_name.empty()) {
      MCTS::MatrixUCBParams<std::remove_cvref_t<decltype(params)>>
          matrix_ucb_params{params};
      // matrix_ucb_params.bandit_params = params;
      matrix_ucb_params.delay = 1 << 12;
      matrix_ucb_params.interval = 64;
      matrix_ucb_params.c = 1.0;
      return search.run(dur, matrix_ucb_params, node, model, battle_data,
                        output);
    } else {
      return search.run(dur, params, node, model, battle_data, output);
    }
  };

  const auto run_2 = [&](auto dur, auto &model) {
    const auto get = [&nodes](auto &node) -> auto & {
      using Node = std::remove_reference_t<decltype(*node)>;
      if (!node.get()) {
        if (nodes.set) {
          throw std::runtime_error("RuntimeSearch::run(): Wrong node type");
        } else {
          nodes.set = true;
          node = std::make_unique<Node>();
          assert(node.get());
        }
      }
      return *node;
    };

    const auto bandit_name_split = Parse::split(agent.bandit_name, '-');

    if (bandit_name_split.size() < 2) {
      throw std::runtime_error("Could not parse bandit string: " +
                               agent.bandit_name);
      return output;
    }

    const auto &name = bandit_name_split[0];
    const float f1 = std::stof(bandit_name_split[1]);

    if (name == "ucb") {
      UCB::Bandit::Params params{.c = f1};
      return run_3(dur, model, params, get(nodes.ucb));
    } else if (name == "ucb1") {
      UCB1::Bandit::Params params{.c = f1};
      return run_3(dur, model, params, get(nodes.ucb1));
    } else if (name == "pucb") {
      PUCB::Bandit::Params params{.c = f1};
      return run_3(dur, model, params, get(nodes.pucb));
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
      return run_3(dur, model, params, get(nodes.exp3));
    } else if (name == "pexp3") {
      PExp3::Bandit::Params params{.gamma = f1,
                                   .one_minus_gamma = (1 - f1),
                                   .alpha = alpha,
                                   .one_minus_alpha = (1 - alpha)};
      return run_3(dur, model, params, get(nodes.pexp3));
    } else {
      throw std::runtime_error("Could not parse bandit string: " + name);
      return output;
    }
  };

  // get model
  const auto search_1 = [&](const auto dur) {
    if (agent.uses_network()) {
      if (!agent.network.has_value()) {
        agent.initialize_network(battle_data.battle);
      }
      return run_2(dur, agent.network.value());
    } else if (agent.network_path == "fp") {
      PokeEngine::Model model{};
      return run_2(dur, model);
    } else {
      MCTS::MonteCarlo model{std::random_device{}()};
      return run_2(dur, model);
    }
  };

  // get duration
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
      return output;
    }
  };

  return search_0();
}

} // namespace RuntimeSearch