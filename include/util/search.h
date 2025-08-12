#pragma once

#include <nn/network.h>
#include <search/bandit/exp3-policy.h>
#include <search/bandit/exp3.h>
#include <search/bandit/ucb-policy.h>
#include <search/bandit/ucb.h>
#include <search/mcts.h>

#include <filesystem>
#include <fstream>
#include <optional>

namespace RuntimeSearch {

struct Nodes {
  std::unique_ptr<Tree::Node<Exp3::JointBandit, MCTS::Obs>> exp3;
  std::unique_ptr<Tree::Node<PExp3::JointBandit, MCTS::Obs>> pexp3;
  std::unique_ptr<Tree::Node<UCB::JointBandit, MCTS::Obs>> ucb;
  std::unique_ptr<Tree::Node<PUCB::JointBandit, MCTS::Obs>> pucb;

  Nodes() { reset(); }

  void reset() {
    exp3 = std::make_unique<decltype(exp3)::element_type>();
    pexp3 = std::make_unique<decltype(pexp3)::element_type>();
    ucb = std::make_unique<decltype(ucb)::element_type>();
    pucb = std::make_unique<decltype(pucb)::element_type>();
  }
};

struct Agent {
  std::string search_time;
  std::string bandit_name;
  std::string network_path;
  // valid if already loaded/cache set
  std::optional<NN::Network> network;
  bool *flag;
};

auto run(BattleData &battle_data, Nodes &nodes, Agent &agent,
         MCTS::Output output = {}) {

  const auto run_2 = [&](auto dur, auto &model) {
    MCTS search{};
    const auto &lower = agent.bandit_name;
    // std::transform(lower.begin(), lower.end(), lower.begin(),
    //                [](auto c) { return std::tolower(c); });

    if (lower.starts_with("exp3-")) {
      const float gamma = std::stof(lower.substr(5));
      Exp3::Bandit::Params params{gamma};
      return search.run(dur, params, *nodes.exp3, battle_data, model, output);
    } else if (lower.starts_with("ucb-")) {
      const float c = std::stof(lower.substr(4));
      UCB::Bandit::Params params{c};
      return search.run(dur, params, *nodes.ucb, battle_data, model, output);
    } else if (lower.starts_with("pexp3-")) {
      const float gamma = std::stof(lower.substr(6));
      PExp3::Bandit::Params params{gamma};
      return search.run(dur, params, *nodes.pexp3, battle_data, model, output);
    } else if (lower.starts_with("pucb-")) {
      const float c = std::stof(lower.substr(5));
      PUCB::Bandit::Params params{c};
      return search.run(dur, params, *nodes.pucb, battle_data, model, output);
    } else {
      throw std::runtime_error("Could not parse bandit string: " + lower);
      return output;
    }
  };

  // get model
  const auto search_1 = [&](const auto dur) {
    if (agent.network_path.empty() || agent.network_path == "mc" ||
        agent.network_path == "montecarlo" ||
        agent.network_path == "monte-carlo") {
      MonteCarlo::Model model{std::random_device{}()};
      return run_2(dur, model);
    } else {
      if (!agent.network.has_value()) {
        agent.network.emplace();
        std::ifstream file{std::filesystem::path{agent.network_path}};
        if (file.fail() || !agent.network.value().read_parameters(file)) {
          throw std::runtime_error("Could not read network params.");
          return MCTS::Output{};
        }
        agent.network.value().fill_cache(battle_data.battle);
      }
      return run_2(dur, agent.network.value());
    }
  };

  // get duration
  const auto search_0 = [&]() {
    if (agent.flag != nullptr) {
      return search_1(agent.flag);
    }
    const auto pos = agent.search_time.find_first_not_of("0123456789");
    size_t number = std::stoll(agent.search_time.substr(0, pos));
    std::string unit = (pos == std::string::npos) ? "" : agent.search_time.substr(pos);
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