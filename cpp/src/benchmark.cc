#include <teams/benchmark-teams.h>
#include <util/argparse.h>
#include <util/random.h>
#include <util/search.h>

struct ProgramArgs : public BenchmarkArgs {};

int benchmark(int argc, char **argv) {

  auto args = argparse::parse<ProgramArgs>(argc, argv);

  auto agent_params = RuntimeSearch::AgentParams{
      .search_budget = args.search_budget.value_or(std::to_string(1 << 20)),
      .bandit = args.bandit.value_or("ucb-1.0"),
      .eval = args.eval.value_or("mc"),
      .matrix_ucb = args.matrix_ucb.value_or(""),
      .discrete = args.use_discrete,
      .table = args.use_table};

  auto agent = RuntimeSearch::Agent{agent_params};

  const uint32_t seed = 1111111;
  auto device = mt19937{seed};
  auto p1 = Teams::benchmark_teams[0];
  auto p2 = Teams::benchmark_teams[1];
  auto battle = PKMN::battle(p1, p2, seed);
  auto options = PKMN::options();
  const auto result = PKMN::update(battle, 0, 0, options);
  const auto durations = PKMN::durations();
  const auto input = MCTS::Input{battle, durations, result};
  auto nodes = RuntimeSearch::Nodes{};

  const auto output = RuntimeSearch::run(device, input, nodes, agent);

  std::cout << output.duration.count() << "µs." << std::endl;
  std::cout << output.iterations << " iterations." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }
