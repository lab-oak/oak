#include <teams/benchmark-teams.h>
#include <util/argparse.h>
#include <util/search.h>

struct ProgramArgs : public AgentArgs {
  std::string &search_time =
      kwarg("search-time", "").set_default(std::to_string(1ULL << 20));
};

int benchmark(int argc, char **argv) {

  auto args = argparse::parse<ProgramArgs>(argc, argv);

  auto agent = RuntimeSearch::Agent{.search_time = args.search_time,
                                    .bandit_name = args.bandit_name,
                                    .network_path = args.network_path,
                                    .discrete_network = args.use_discrete,
                                    .matrix_ucb_name = args.matrix_ucb_name};

  auto nodes = RuntimeSearch::Nodes{};

  auto p1 = Teams::benchmark_teams[0];
  auto p2 = Teams::benchmark_teams[1];

  const uint64_t seed = 1111111;

  auto battle = PKMN::battle(p1, p2, seed);
  auto options = PKMN::options();
  const auto result = PKMN::update(battle, 0, 0, options);
  const auto durations = PKMN::durations();
  MCTS::BattleData battle_data{battle, durations, result};

  const auto output = RuntimeSearch::run(battle_data, nodes, agent);

  std::cout << output.duration.count() << " ms." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }
