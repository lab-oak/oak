#pragma once

#include <util/vector.h>

#include <data/strings.h>

#include <array>
#include <bit>
#include <cstdint>
#include <functional>
#include <iostream>
#include <unordered_map>

#include <ii/random-battles/random-set-data.h>

// WIP clone of the official showdown random team generator
namespace RandomBattles {

using PartialSet = Data::OrderedMoveSet;

struct PartialTeam {
  using SpeciesSlot = std::pair<Data::Species, uint8_t>;
  std::array<SpeciesSlot, 6> species_slots;
  std::array<PartialSet, 6> move_sets;

  void sort() {
    std::sort(species_slots.begin(), species_slots.end(),
              [](const auto &slot1, const auto &slot2) {
                return slot1.first > slot2.first;
              });
  }

  bool matches(const PartialTeam &complete) const {
    int j = 0;
    for (int i = 0; i < 6; ++i) {

      const auto smaller_species = species_slots[i].first;
      if (smaller_species == Data::Species::None) {
        break;
      }

      while (j < 6) {
        if (species_slots[i].first == complete.species_slots[j].first) {
          const bool matches =
              complete.move_sets[complete.species_slots[j].second].contains(
                  move_sets[species_slots[i].second]);
          if (!matches) {
            return false;
          }
          break;
        }
        if (complete.species_slots[j].first < species_slots[i].first ||
            (j == 5)) {
          return false;
        }
        ++j;
      }
    }
    return true;
  }

  void print() const noexcept {
    for (int i = 0; i < 6; ++i) {
      const auto pair = species_slots[i];
      if (pair.first == Data::Species::None) {
        continue;
      }
      std::cout << Names::species_string(pair.first) << ": ";
      const auto &set_data = move_sets[pair.second]._data;
      for (int m = 0; m < 4; ++m) {
        if (set_data[m] == Data::Moves::None) {
          continue;
        }
        std::cout << Names::move_string(set_data[m]) << ", ";
      }
      std::cout << std::endl;
    }
  }
};

using Seed = int64_t;

namespace PRNG2 {

constexpr void next(Seed &seed) noexcept {
  static constexpr int64_t a{0x5D588B656C078965};
  static constexpr int64_t c{0x0000000000269EC3};
  seed = a * seed + c;
}

constexpr auto next(Seed &seed, auto to) noexcept {
  next(seed);
  const uint32_t top = seed >> 32;
  return to * ((double)top / 0x100000000);
}

constexpr auto next(Seed &seed, auto from, auto to) noexcept {
  next(seed);
  const uint32_t top = seed >> 32;
  return from + (to - from) * ((double)top / 0x100000000);
}

template <typename Container>
void shuffle(Seed &seed, Container &items) noexcept {
  auto start = 0;
  const auto end = items.size();
  while (start < end - 1) {
    const auto nextIndex = next(start, end);
    if (start != nextIndex) {
      std::swap(items[start], items[nextIndex]);
    }
    ++start;
  }
}

}; // namespace PRNG2

struct PRNG {
  int64_t seed;

  PRNG(int64_t seed) : seed{seed} {}

  int64_t nextFrame(int64_t seed) {
    static constexpr int64_t a{0x5D588B656C078965};
    static constexpr int64_t c{0x0000000000269EC3};
    seed = a * seed + c;
    return seed;
  }

  void next() noexcept { seed = nextFrame(seed); }

  int next(int to) noexcept {
    seed = nextFrame(seed);
    const uint32_t top = seed >> 32;
    return to * ((double)top / 0x100000000);
  }

  int next(int from, int to) {
    seed = nextFrame(seed);
    const uint32_t top = seed >> 32; // Use the upper 32 bits
    return from + (to - from) * ((double)top / 0x100000000);
  }

  template <typename Container>
  void shuffle(Container &items, int start = 0, int end = -1) noexcept {
    if (end < 0) {
      end = items.size();
    }
    while (start < end - 1) {
      const auto nextIndex = next(start, end);
      if (start != nextIndex) {
        std::swap(items[start], items[nextIndex]);
      }
      ++start;
    }
  }

  template <typename Container, typename T>
  const T &sample(const Container &items) {
    assert(items.size() > 0);
    return items[next(items.size())];
  }

  bool randomChance(int numerator, int denominator) {
    return next(denominator) < numerator;
  }

  void display() {
    const auto *data = reinterpret_cast<uint16_t *>(&seed);
    std::cout << "[ ";
    for (int i = 3; i >= 1; --i) {
      std::cout << (int)data[i] << ", ";
    }
    std::cout << (int)data[0] << " ]\n";
  }
};

template <typename Container>
auto fastPop(Container &container, auto index) noexcept {
  const auto n = container.size();
  const auto val = container[index];
  container[index] = container[n - 1];
  container.resize(n - 1);
  return val;
}

template <typename Container>
auto sampleNoReplace(Container &container, PRNG &prng) noexcept {
  const auto n = container.size();
  const auto index = prng.next(n);
  return fastPop<Container>(container, index);
}

class Teams {
private:
  PRNG prng;
  bool battleHasDitto = false;

public:
  Teams(PRNG prng) : prng{prng} {}

  PartialTeam randomTeam() {

    PartialTeam team{};
    auto n_pokemon = 0;

    prng.next(); // for type sample call

    std::array<int, 15> typeCount{};
    std::array<int, 6> weaknessCount{};
    int numMaxLevelPokemon{};

    using Arr = ArrayBasedVector<146>::Vector<Data::Species>;

    Arr pokemonPool{RandomBattlesData::pokemonPool};
    Arr rejectedButNotInvalidPool{};

    while (n_pokemon < 6 && pokemonPool.size()) {
      auto species = sampleNoReplace(pokemonPool, prng);

      if (species == Data::Species::Ditto && battleHasDitto) {
        continue;
      }

      bool skip = false;

      // types
      const auto types = Data::get_types(species);
      for (const Data::Types type : types) {
        if (typeCount[static_cast<uint8_t>(type)] >= 2) {
          skip = true;
          break;
        }
      }
      if (skip) {
        rejectedButNotInvalidPool.push_back(species);
        continue;
      }

      // weakness
      const auto &w =
          RandomBattlesData::RANDOM_SET_DATA[static_cast<uint8_t>(species)]
              .weaknesses;
      for (int i = 0; i < 6; ++i) {
        if (!w[i]) {
          continue;
        }
        if (weaknessCount[i] >= 2) {
          skip = true;
          break;
        }
      }
      if (skip) {
        rejectedButNotInvalidPool.push_back(species);
        continue;
      }

      // lvl 100
      if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1) {
        skip = true;
      }
      if (skip) {
        rejectedButNotInvalidPool.push_back(species);
        continue;
      }

      // accept the set
      team.species_slots[n_pokemon] = {species, n_pokemon};
      team.move_sets[n_pokemon++] = randomSet(species);

      // update
      typeCount[static_cast<uint8_t>(types[0])]++;
      if (types[0] != types[1]) {
        typeCount[static_cast<uint8_t>(types[1])]++;
      }

      for (int i = 0; i < 6; ++i) {
        weaknessCount[i] += w[i];
      }
      if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1) {
        ++numMaxLevelPokemon;
      }
      if (species == Data::Species::Ditto) {
        battleHasDitto = true;
      }
    }

    while (n_pokemon < 6 && rejectedButNotInvalidPool.size()) {
      const auto species = sampleNoReplace(rejectedButNotInvalidPool, prng);
      team.species_slots[n_pokemon] = {species, n_pokemon};
      team.move_sets[n_pokemon++] = randomSet(species);
    }

    team.sort();
    return team;
  }

  PartialSet randomSet(Data::Species species) {
    PartialSet set{};
    auto set_size = 0;

    const auto data{
        RandomBattlesData::RANDOM_SET_DATA[static_cast<int>(species)]};
    constexpr auto maxMoveCount = 4;

    // combo moves
    if (data.n_combo_moves && (data.n_combo_moves <= maxMoveCount) &&
        prng.randomChance(1, 2)) {
      for (int m = 0; m < data.n_combo_moves; ++m) {
        set_size += set.insert(data.combo_moves[m]);
      }
    }

    // exclusive moves
    if ((set_size < maxMoveCount) && data.n_exclusive_moves) {
      set_size +=
          set.insert(data.exclusive_moves[prng.next(data.n_exclusive_moves)]);
    }

    // essential moves
    if ((set_size < maxMoveCount) && data.n_essential_moves) {
      for (int m = 0; m < data.n_essential_moves; ++m) {
        set_size += set.insert(data.essential_moves[m]);
        if (set_size == maxMoveCount) {
          break;
        }
      }
    }

    ArrayBasedVector<RandomBattlesData::RandomSetEntry::max_moves>::Vector<
        Data::Moves>
        movePool{data.moves};
    movePool.resize(data.n_moves);
    while ((set_size < maxMoveCount) && movePool.size()) {
      set_size += set.insert(sampleNoReplace(movePool, prng));
    }

    assert(set_size == maxMoveCount);

    prng.shuffle(set._data);
    // sort before returning for fast comparison
    set.sort();

    return set;
  }

  PartialSet finishSet(const Data::Species species, const PartialSet &set) {
    while (true) {
      const auto finished = randomSet(species);
      if (finished.contains(set)) {
        return finished;
      }
    }
  }

  void finishTeam(PartialTeam &partial) {
    using Arr = ArrayBasedVector<146>::Vector<Data::Species>;

    //   const bool species_done = !std::any();

    //   if (!species_done) {
    //     // get valid species, add to partial
    //     Arr pokemonPool{RandomBattlesData::pokemonPool};
    //     Arr validPool{}
    //   }

    //   for (auto p = 0; p < 6; ++p) {
    //     const auto pair = partial.species_slots[p];
    //     const auto& set = partial.move_sets[pair.second];
    //     if (std::any()) {
    //       set = finishSet(set);
    //     }
    //   }
    //   partial.sort();
  }
};

} // namespace RandomBattles
