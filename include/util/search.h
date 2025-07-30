#pragma once

#include <nn/network.h>
#include <search/mcts.h>

#include <filesystem>
#include <fstream>

namespace RuntimeSearch {

template <typename... Bandits>
auto run(const BattleData &battle_data, size_t count, char mode,
         std::string bandit_name, std::string network_path) {

  const auto run_2 = [&](auto dur, auto &model) {
    MCTS search{};
    return search.go<Bandits...>(bandit_name, dur, battle_data, model);
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
      return run_2(dur, network);
    }
  };

  const auto run_0 = [&]() {
    if (mode == 'i') {
      return run_1(count);
    } else if (mode == 't') {
      return run_1(std::chrono::milliseconds{count});
    } else {
      throw std::runtime_error("Invalid duration mode char.");
      return MCTS::Output{};
    }
  };

  return run_0();
}

} // namespace RuntimeSearch