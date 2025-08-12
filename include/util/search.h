#pragma once

#include <nn/network.h>
#include <search/exp3-policy.h>
#include <search/exp3.h>
#include <search/mcts.h>
#include <search/ucb-policy.h>
#include <search/ucb.h>

#include <filesystem>
#include <fstream>

namespace RuntimeSearch {

auto run(const BattleData &battle_data, size_t count, char mode,
         std::string bandit_name, std::string network_path) {

  const auto run_2 = [&](auto dur, auto &model) {
    MCTS search{};
    auto lower = bandit_name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](auto c) { return std::tolower(c); });

    if (lower.starts_with("exp3-")) {
      const float gamma = std::stof(lower.substr(5));
      Exp3::Bandit::Params params{gamma};
      Tree::Node<Exp3::JointBandit, MCTS::Obs> node{};
      return search.run(dur, params, node, battle_data, model);
    } else if (lower.starts_with("ucb-")) {
      const float c = std::stof(lower.substr(4));
      UCB::Bandit::Params params{c};
      Tree::Node<UCB::JointBandit, MCTS::Obs> node{};
      return search.run(dur, params, node, battle_data, model);
    } else if (lower.starts_with("pexp3-")) {
      const float gamma = std::stof(lower.substr(6));
      PExp3::Bandit::Params params{gamma};
      Tree::Node<PExp3::JointBandit, MCTS::Obs> node{};
      return search.run(dur, params, node, battle_data, model);
    } else if (lower.starts_with("pucb-")) {
      const float c = std::stof(lower.substr(5));
      PUCB::Bandit::Params params{c};
      Tree::Node<PUCB::JointBandit, MCTS::Obs> node{};
      return search.run(dur, params, node, battle_data, model);
    } else {
      throw std::runtime_error("Could not parse bandit string: " + lower);
    }
  };

  const auto run_1 = [&](auto dur) {
    if (network_path == "mc") {
      MonteCarlo::Model model{};
      return run_2(dur, model);
    } else {
      std::ifstream file{std::filesystem::path{network_path}};
      NN::Network network{};
      if (file.fail() || !network.read_parameters(file)) {
        throw std::runtime_error("Could not read network params.");
        return MCTS::Output{};
      }
      network.fill_cache(battle_data.battle);
      return run_2(dur, network);
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
      throw std::runtime_error("Invalid duration mode char.");
      return MCTS::Output{};
    }
  };

  return run_0();
}

} // namespace RuntimeSearch