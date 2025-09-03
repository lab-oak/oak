#include <format/OU/teams.h>
#include <search/bandit/exp3.h>
#include <search/mcts.h>
#include <search/tree.h>
#include <util/random.h>

struct FastModel {
  fast_prng device;

  FastModel(uint8_t *data) : device{data} {}
};

int benchmark(int argc, char **argv) {

  auto p1 = Format::Data::benchmark_teams[0];
  auto p2 = Format::Data::benchmark_teams[1];

  const uint64_t seed = 1111111;

  auto battle = PKMN::battle(p1, p2, seed);
  auto options = PKMN::options();
  const auto result = PKMN::update(battle, 0, 0, options);
  const auto durations = PKMN::durations();
  MCTS::BattleData battle_data{battle, durations, result};

  FastModel model{battle_data.battle.bytes +
                  PKMN::Layout::Offsets::Battle::rng};

  MCTS::Search search{};
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

  Exp3::Bandit::Params bandit_params{.03};
  using Node = Tree::Node<Exp3::JointBandit, MCTS::Obs>;
  Node node{};
  const auto output =
      search.run(iterations, bandit_params, node, model, battle_data);

  std::cout << output.duration.count() << " ms." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }
