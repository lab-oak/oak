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
#include <memory>
#include <thread>
#include <variant>

namespace RuntimeSearch {

template <typename T> using UniqueNode = std::unique_ptr<MCTS::Node<T>>;
template <typename T> using UniqueTable = std::unique_ptr<MCTS::Table<T>>;

// TODO templatize
using BanditVariant =
    std::variant<std::monostate, UniqueNode<Exp3::JointBandit>,
                 UniqueTable<Exp3::JointBandit>, UniqueNode<PExp3::JointBandit>,
                 UniqueTable<PExp3::JointBandit>, UniqueNode<UCB::JointBandit>,
                 UniqueTable<UCB::JointBandit>, UniqueNode<PUCB::JointBandit>,
                 UniqueTable<PUCB::JointBandit>, UniqueNode<UCB1::JointBandit>,
                 UniqueTable<UCB1::JointBandit>>;

struct Heap {
  BanditVariant data;

  bool empty() const noexcept {
    return std::holds_alternative<std::monostate>(data);
  }

  template <typename T> auto both() {
    UniqueNode<T> *a = std::get_if<UniqueNode<T>>(&data);
    UniqueTable<T> *b = std::get_if<UniqueTable<T>>(&data);
    return std::pair<UniqueNode<T> *, UniqueTable<T> *>{a, b};
  }

  // If the variant is a node, it swaps
  bool update(uint8_t i, uint8_t j, MCTS::Obs &obs) {
    const auto lambda = [&](auto &node) {
      using T = std::remove_cvref_t<decltype(node)>;
      if constexpr (TypeTraits::is_node<T>) {
        if (!node || !node->stats.is_init()) {
          return false;
        }
        auto child = node->children.find({i, j, obs});
        if (child == node->children.end()) {
          node = std::make_unique<std::decay_t<decltype(*node)>>();
          return false;
        } else {
          auto unique_child = std::make_unique<std::decay_t<decltype(*node)>>(
              std::move((*child).second));
          node.swap(unique_child);
          return true;
        }
      }
      return true;
    };

    return std::visit(lambda, data);
  }
};

struct AgentParams {
  std::string search_budget;
  std::string bandit;
  std::string eval;
  std::string matrix_ucb;
  bool discrete;
  bool table;

  constexpr bool operator==(const AgentParams &) const = default;
};

struct Agent : AgentParams {

  std::unique_ptr<NN::Battle::NetworkBase> network_ptr{};

  Agent(const AgentParams &params) : AgentParams{params}, network_ptr{} {}
  Agent() = default;
  // Agent(const Agent &other) : AgentParams{static_cast<AgentParams>(other)} {}

  constexpr bool operator==(const Agent &other) const {
    return static_cast<AgentParams>(*this) == static_cast<AgentParams>(other);
  }

  bool is_monte_carlo() const {
    return eval.empty() || eval == "mc" || eval == "montecarlo" ||
           eval == "monte-carlo";
  }
  bool is_foul_play() const { return eval == "fp"; }
  bool is_network() const { return !is_monte_carlo() && !is_foul_play(); }

  void initialize_network(const pkmn_gen1_battle &b) {
    std::cout << "start init: " << network_ptr.get() << std::endl;

    auto network = std::make_unique<NN::Battle::Network>();
    std::cout << "temp float: " << network.get() << std::endl;

    // sometimes reads fail because python is writing to that file. just retry
    const auto try_read_parameters = [&network, this]() {
      constexpr auto tries = 3;
      for (auto i = 0; i < tries; ++i) {
        std::ifstream file{eval};
        if (network->read_parameters(file)) {
          return;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      throw std::runtime_error{"Agent could not read network parameters at: " +
                               eval};
    };
    try_read_parameters();

    // network.init_caches(b);

    if (discrete) {
      const auto [id, hd, vd, pd] = network->main_net.shape();
      std::cout << "temp shape " << id << ' ' << hd << ' ' << vd << ' ' << pd
                << std::endl;
      auto q_network_ptr = NN::Battle::visit_network_or_construct(
          id, hd, vd, pd, [](auto &net) { return; });
      q_network_ptr = NN::Battle::visit_network_or_construct(
          id, hd, vd, pd,
          [&network](auto &net) {
            // TODO call param, cache copy
            std::cout << "visit lambda" << std::endl;
            net.active_net = network->active_net;
            net.pokemon_net = network->pokemon_net;
            net.pokemon_out_dim = network->pokemon_out_dim;
            net.active_out_dim = network->active_out_dim;
            net.side_embedding_dim = network->side_embedding_dim;
            net.battle_embedding.resize(network->battle_embedding.size());
            net.battle_cache = network->battle_cache;
            return;
          },
          std::move(q_network_ptr));
      std::cout << "discrete net " << q_network_ptr.get() << std::endl;
      if (q_network_ptr) {
        std::cout << "before reset" << std::endl;
        network.reset();
        network_ptr = std::move(q_network_ptr);
        std::cout << "member after swap: " << network_ptr.get() << std::endl;
      }
    } else {
      network_ptr = std::move(network);
    }

    std::cout << "end init: " << network_ptr.get() << std::endl;

    assert(network_ptr);
  }
};

auto run(auto &device, const MCTS::Input &input, Heap &heap_variant,
         Agent &agent, MCTS::Output output = {}, bool *const flag = nullptr) {

  const auto parse_eval_and_search = [&](const auto dur, const auto &params,
                                         auto &heap) {
    MCTS::Search s{};
    if (agent.is_monte_carlo()) {
      MCTS::MonteCarlo model{};
      return s.run(device, dur, params, heap, model, input, output);
    } else if (agent.is_foul_play()) {
      PokeEngine::Model model{};
      return s.run(device, dur, params, heap, model, input, output);
    } else {
      if (!agent.network_ptr) {
        agent.initialize_network(input.battle);
      }
      if (auto network =
              dynamic_cast<NN::Battle::Network *>(agent.network_ptr.get());
          network) {
        return s.run(device, dur, params, heap, *network, input, output);
      } else {
        const auto [id, hd, vd, pd] = agent.network_ptr->shape();
        auto q_network_ptr = NN::Battle::visit_network_or_construct(
            id, hd, vd, pd,
            [&](auto &net) {
              output = s.run(device, dur, params, heap, net, input, output);
            },
            std::move(agent.network_ptr));
        if (q_network_ptr) {
          agent.network_ptr = std::move(q_network_ptr);
        }
        return output;
      }
    }
  };

  const auto parse_heap_and_search = [&](const auto dur, const auto &params,
                                         const auto &both) {
    const auto [node_ptr, table_ptr] = both;
    using Node = std::remove_cvref_t<decltype(**node_ptr)>;
    using Table = std::remove_cvref_t<decltype(**table_ptr)>;
    auto &heap = heap_variant.data;
    if (agent.table) {
      if (heap_variant.empty()) {
        auto table = std::make_unique<Table>();
        table->hasher = {device};
        heap = std::move(table);
        return parse_eval_and_search(dur, params,
                                     *std::get<std::unique_ptr<Table>>(heap));
      } else if (!table_ptr) {
        throw std::runtime_error{"Bad variant access."};
      }
      return parse_eval_and_search(dur, params,
                                   *std::get<std::unique_ptr<Table>>(heap));
    } else {
      if (heap_variant.empty()) {
        heap = std::make_unique<Node>();
        return parse_eval_and_search(dur, params,
                                     *std::get<std::unique_ptr<Node>>(heap));
      } else if (!node_ptr) {
        throw std::runtime_error{"Bad variant access."};
      }
      return parse_eval_and_search(dur, params,
                                   *std::get<std::unique_ptr<Node>>(heap));
    }
  };

  const auto parse_matrix_ucb_and_search = [&](auto dur, auto &bandit_params,
                                               const auto &both) {
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
      return parse_heap_and_search(dur, matrix_ucb_params, both);
    } else {
      return parse_heap_and_search(dur, bandit_params, both);
    }
  };

  const auto parse_bandit_and_search = [&](auto dur) {
    const auto bandit_split = Parse::split(agent.bandit, '-');
    if (bandit_split.size() < 2) {
      throw std::runtime_error("Could not parse bandit string: " +
                               agent.bandit);
    }

    const auto &name = bandit_split[0];
    const float f1 = std::stof(bandit_split[1]);
    if (name == "ucb") {
      UCB::Bandit::Params params{.c = f1};
      return parse_matrix_ucb_and_search(dur, params,
                                         heap_variant.both<UCB::JointBandit>());
    } else if (name == "ucb1") {
      UCB1::Bandit::Params params{.c = f1};
      return parse_matrix_ucb_and_search(
          dur, params, heap_variant.both<UCB1::JointBandit>());
    } else if (name == "pucb") {
      PUCB::Bandit::Params params{.c = f1};
      return parse_matrix_ucb_and_search(
          dur, params, heap_variant.both<PUCB::JointBandit>());
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
      return parse_matrix_ucb_and_search(
          dur, params, heap_variant.both<Exp3::JointBandit>());
    } else if (name == "pexp3") {
      PExp3::Bandit::Params params{.gamma = f1,
                                   .one_minus_gamma = (1 - f1),
                                   .alpha = alpha,
                                   .one_minus_alpha = (1 - alpha)};
      return parse_matrix_ucb_and_search(
          dur, params, heap_variant.both<PExp3::JointBandit>());
    } else {
      throw std::runtime_error("Could not parse bandit string: " + name);
    }
  };

  const auto parse_budget_and_search = [&]() {
    if (flag != nullptr) {
      return parse_bandit_and_search(flag);
    }
    const auto pos = agent.search_budget.find_first_not_of("0123456789");
    size_t number = std::stoll(agent.search_budget.substr(0, pos));
    std::string unit =
        (pos == std::string::npos) ? "" : agent.search_budget.substr(pos);
    if (unit.empty()) {
      return parse_bandit_and_search(number);
    } else if (unit == "ms" || unit == "millisec" || unit == "milliseconds") {
      return parse_bandit_and_search(std::chrono::milliseconds{number});
    } else if (unit == "s" || unit == "sec" || unit == "seconds") {
      return parse_bandit_and_search(std::chrono::seconds{number});
    } else {
      throw std::runtime_error("Invalid search duration specification: " +
                               agent.search_budget);
    }
  };

  return parse_budget_and_search();
}

} // namespace RuntimeSearch