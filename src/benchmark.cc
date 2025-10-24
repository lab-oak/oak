#include <search/bandit/exp3.h>
#include <search/mcts.h>
#include <search/tree.h>
#include <teams/benchmark-teams.h>
#include <util/random.h>
#include <util/search.h>

struct FastModel {
  fast_prng device;

  FastModel(uint8_t *data) : device{data} {}
};

int benchmark(int argc, char **argv) {

  auto p1 = Teams::benchmark_teams[0];
  auto p2 = Teams::benchmark_teams[1];

  const uint64_t seed = 1111111;

  auto battle = PKMN::battle(p1, p2, seed);
  auto options = PKMN::options();
  const auto result = PKMN::update(battle, 0, 0, options);
  const auto durations = PKMN::durations();
  MCTS::BattleData battle_data{battle, durations, result};

  int exp = 20;
  std::string bandit_name{"exp3-0.03"};
  std::string network_path{"mc"};

  if (argc > 1) {
    exp = std::atoi(argv[1]);
  }
  if (argc > 2) {
    bandit_name = {argv[2]};
  }
  if (argc > 3) {
    network_path = {argv[3]};
  }

  exp = std::max(0, std::min(20, exp));
  std::cout << "Benchmarking for 2^" << exp << " iterations." << std::endl;
  const size_t iterations = 1 << exp;

  RuntimeSearch::Nodes nodes{};
  RuntimeSearch::Agent agent{std::to_string(iterations), bandit_name,
                             network_path};

  const auto output = RuntimeSearch::run(battle_data, nodes, agent);

  std::cout << output.duration.count() << " ms." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }
