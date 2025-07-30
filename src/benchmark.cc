#include <data/sample-teams.h>
#include <search/exp3.h>
#include <search/mcts.h>
#include <search/tree.h>
#include <search/ucb.h>
#include <util/random.h>

#include <chrono>

struct FastModel {
  fast_prng device;

  FastModel(uint8_t *data) : device{data} {}
};

int benchmark(int argc, char **argv) {
  constexpr bool node_visits = false;

  auto p1 = SampleTeams::benchmark_teams[0];
  auto p2 = SampleTeams::benchmark_teams[1];

  const uint64_t seed = 1111111;
  // MonteCarlo::Model model{mt19937{seed}};
  const auto battle = PKMN::battle(p1, p2, seed);
  const auto durations = PKMN::durations(p1, p2);
  BattleData battle_data{battle, durations};

  FastModel model{battle_data.battle.bytes + Layout::Offsets::Battle::rng};
  MCTS search{};
  int exp = 20;
  std::string bandit_string{"exp3"};
  if (argc > 1) {
    exp = std::atoi(argv[1]);
  }
  if (argc > 2) {
    bandit_string = {argv[2]};
  }

  exp = std::max(0, std::min(20, exp));
  std::cout << "Benchmarking for 2^" << exp << " iterations." << std::endl;
  const size_t iterations = 1 << exp;

  battle_data.result = PKMN::update(battle_data.battle, 0, 0, search.options);

  const auto output = search.go<Exp3::JointBanditData<.03f, node_visits>,
                                UCB::JointBanditData<2.0f>>(
      bandit_string, iterations, battle_data, model);
  std::cout << output.duration.count() << " ms." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }
