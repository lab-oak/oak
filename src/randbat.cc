#include <map>
#include <mutex>
#include <random>
#include <thread>

#include <ii/random-battles/randbat.h>

int benchmark(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "enter number of teams to generate" << std::endl;
    return 1;
  }

  RandomBattles::PRNG prng{32049283049280};
  for (int i = 0; i < std::atoi(argv[1]); ++i) {
    prng.next();
    RandomBattles::Teams generator{prng};
    generator.randomTeam();
  }
  return 0;
}

int generate_team(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "Usage: Enter 4 numbers representing the pokemon-showdown "
                 "seed; Recieve team"
              << std::endl;
    return 1;
  }

  uint16_t seed_raw[4];
  seed_raw[3] = std::atoi(argv[1]);
  seed_raw[2] = std::atoi(argv[2]);
  seed_raw[1] = std::atoi(argv[3]);
  seed_raw[0] = std::atoi(argv[4]);
  int64_t seed = *reinterpret_cast<int64_t *>(seed_raw);

  std::cout << "seed: ";
  for (int i = 0; i < 4; ++i) {
    std::cout << seed_raw[3 - i] << ' ';
  }
  std::cout << " = " << seed << std::endl;

  RandomBattles::Teams generator{RandomBattles::PRNG{seed}};
  const auto team = generator.randomTeam();
  team.print();
  return 0;
}

struct Sync {
  std::mutex mutex;
  struct Data {
    size_t n{};
    RandomBattles::PartialTeam team;
  };
  std::unordered_map<int64_t, Data> map;
};

template <typename Sync>
void sample_teams(Sync *sync, const RandomBattles::PartialTeam partial) {

  while (true) {
    std::mt19937_64 rng{};
    rng.seed(std::random_device{}());
    int64_t seed = rng();
    RandomBattles::Teams generator{RandomBattles::PRNG{seed}};
    const auto team = generator.randomTeam();
    if (partial.matches(team)) {
      sync->mutex.lock();
      auto &data = sync->map[seed];
      data.team = team;
      ++data.n;
      if (data.n == 1) {
        std::cout << "\nseed: " << seed << std::endl;
        team.print();
      }
      sync->mutex.unlock();
    }
  }
}

int try_generate_all_sets(int argc, char **argv) {
  std::mt19937_64 rng{};
  rng.seed(std::random_device{}());
  int64_t initial_seed = rng();
  RandomBattles::PRNG prng{initial_seed};

  const size_t tries_per_species = 10000;
  // using Map = LinearScanMap<RandomBattles::PartialSet, bool>;
  using Map = std::map<RandomBattles::PartialSet, bool>;
  constexpr auto n_species = RandomBattlesData::pokemonPool.size();
  std::array<Map, n_species> sets{};
  auto total_sets = 0;

  for (int i = 0; i < n_species; ++i) {
    const auto species = RandomBattlesData::pokemonPool[i];
    for (int t = 0; t < tries_per_species; ++t) {
      auto generator = RandomBattles::Teams{prng};
      auto set = generator.randomSet(species);
      set.sort();
      sets[i][set] = true;
      prng.next();
    }
    total_sets += sets[i].size();
    std::cout << Names::species_string(species) << " : " << sets[i].size()
              << std::endl;
    for (const auto &pair : sets[i]) {
      const auto &arr = pair.first._data;
      for (const auto move : arr) {
        std::cout << Names::move_string(move) << "(" << static_cast<int>(move)
                  << "), ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << "Total Sets: " << total_sets << std::endl;
  return 0;
}

int main(int argc, char **argv) { return generate_team(argc, argv); }
