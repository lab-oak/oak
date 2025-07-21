#include <battle/sample-teams.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>
#include <pi/ucb.h>
#include <util/random.h>

#include <chrono>

struct FastModel {
  fast_prng device;

  FastModel(uint8_t *data) : device{data} {}
};

int benchmark(int argc, char **argv) {
  constexpr bool node_visits = false;

  using Obs = std::array<uint8_t, 16>;
  using Exp3Node = Tree::Node<Exp3::JointBanditData<.03f, node_visits>, Obs>;
  using UCBNode = Tree::Node<UCB::JointBanditData, Obs>;

  // using Table = Tree::Table<Exp3::JointBanditData<.03f, node_visits>, Obs,
  //                           1 << 21, uint32_t>;
  // auto &table = *new Table{};

  auto p1 = SampleTeams::benchmark_teams[0];
  auto p2 = SampleTeams::benchmark_teams[1];

  const uint64_t seed = 1111111;
  // MonteCarlo::Model model{prng{seed}};
  const auto battle = PKMN::battle(p1, p2, seed);
  const auto durations = PKMN::durations(p1, p2);
  BattleData battle_data{battle, durations};

  FastModel model{battle_data.battle.bytes + Layout::Offsets::Battle::rng};
  MCTS search{};
  int exp = 20;
  if (argc == 2) {
    exp = std::atoi(argv[1]);
  }
  exp = std::max(0, std::min(20, exp));
  size_t iterations = 1 << exp;
  battle_data.result = PKMN::update(battle_data.battle, 0, 0, search.options);
  Exp3Node node{};

  const auto output = search.run(iterations, node, battle_data, model);
  std::cout << output.duration.count() << " ms." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }
