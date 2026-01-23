#pragma once

#include <format/random-battles/random-set-data.h>
#include <libpkmn/pkmn.h>

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <unordered_map>

// Provides minimal std::vector interface around a std::array
// May offer better performance in some cases
template <std::size_t max_size> struct ArrayBasedVector {
  template <typename T, typename CapacityT = std::size_t> class Vector {
  protected:
    std::array<T, max_size> _storage;
    CapacityT _size;

  public:
    constexpr Vector() : _size{} {}

    template <typename InT>
      requires(std::is_integral_v<InT>)
    constexpr Vector(const InT n) {
      assert(0 < n && n <= max_size);
      _size = n;
      std::fill(this->begin(), this->end(), T{});
    }

    template <typename Vec> constexpr Vector(const Vec &other) noexcept {
      assert(other.size() <= max_size);
      _size = other.size();
      std::copy(other.begin(), other.end(), _storage.begin());
    }

    template <typename Vec>
    constexpr Vector &operator=(const Vec &other) noexcept {
      assert(other.size() <= max_size);
      _size = other.size();
      std::copy(other.begin(), other.end(), _storage.begin());
      return *this;
    }

    template <typename Vec> bool operator==(const Vec &other) const noexcept {
      for (CapacityT i = 0; i < _size; ++i) {
        if ((*this)[i] != other[i]) {
          return false;
        }
      }
      return _size == other.size();
    }

    template <typename size_type>
    constexpr void resize(size_type n, T val = T{}) {
      assert(n <= max_size);
      if (_size < n) {
        std::fill(_storage.begin() + _size, _storage.begin() + n, val);
      }
      _size = n;
    }

    template <typename size_type> void reserve(size_type n) noexcept {
      assert(n <= max_size);
      _size = n;
    }

    constexpr void push_back(const T &val = T{}) {
      assert(_size < max_size);
      _storage[_size++] = val;
    }

    constexpr void push_back(T &&val = T{}) {
      assert(_size < max_size);
      _storage[_size++] = val;
    }

    constexpr T &operator[](auto n) { return _storage[n]; }

    constexpr const T &operator[](auto n) const { return _storage[n]; }

    CapacityT size() const noexcept { return _size; }

    constexpr void clear() noexcept { _size = 0; }

    constexpr auto begin() noexcept { return _storage.begin(); }

    constexpr const auto begin() const noexcept { return _storage.begin(); }

    constexpr auto end() noexcept { return _storage.begin() + _size; }

    const auto end() const noexcept { return _storage.begin() + _size; }
  };
};

namespace Detail {
template <typename T, size_t n>
  requires(std::is_enum_v<T>)
class OrderedArrayBasedSet {
public:
  std::array<T, n> _data;

public:
  void sort() noexcept {
    std::sort(_data.begin(), _data.end(), std::greater<T>());
  }

  // OrderedArrayBasedSet& operator=(const OrderedArrayBasedSet&) = default;
  bool operator==(const OrderedArrayBasedSet &) const noexcept = default;

  bool operator<(const OrderedArrayBasedSet &other) const noexcept {
    return _data < other._data;
  }

  bool insert(const T &val) noexcept {
    auto free_index = -1;
    bool is_in = false;
    for (auto i = 0; i < n; ++i) {
      if (_data[i] == val) {
        return false;
      }
      if (_data[i] == T{0}) {
        free_index = i;
      }
    }
    if (free_index >= 0) {
      _data[free_index] = val;
      return true;
    }
    return false;
  }

  bool contains(const OrderedArrayBasedSet &other) const noexcept {
    int i = 0;
    for (int other_i = 0; other_i < 4; ++other_i) {
      const auto other_move = other._data[other_i];
      if (other_move == T{0}) {
        break;
      }
      while (true) {
        if (other_move == _data[i]) {
          ++i;
          break;
        } else {
        }
        if ((i >= 4) || (other_move > _data[i])) {
          return false;
        }
        ++i;
      }
    }
    return true;
  }

  const auto begin() const { return _data.begin(); }

  const auto end() const { return _data.end(); }
};
}; // namespace Detail

using OrderedMoveSet = Detail::OrderedArrayBasedSet<PKMN::Data::Move, 4>;

// WIP clone of the official showdown random team generator
namespace RandomBattles {

using PKMN::Data::Move;
using PKMN::Data::Species;
using PKMN::Data::Type;

using PartialSet = OrderedMoveSet;

struct PartialTeam {
  using SpeciesSlot = std::pair<Species, uint8_t>;
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
      if (smaller_species == Species::None) {
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
      if (pair.first == Species::None) {
        continue;
      }
      std::cout << PKMN::species_string(pair.first) << ": ";
      const auto &set_data = move_sets[pair.second]._data;
      for (int m = 0; m < 4; ++m) {
        if (set_data[m] == Move::None) {
          continue;
        }
        std::cout << PKMN::move_string(set_data[m]) << ", ";
      }
      std::cout << std::endl;
    }
  }
};

using Seed = int64_t;

namespace PRNG2 {

constexpr void next(Seed &seed) noexcept {
  constexpr int64_t a{0x5D588B656C078965};
  constexpr int64_t c{0x0000000000269EC3};
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

  PKMN::Team partialToTeam(const PartialTeam &partial) const {
    PKMN::Team team{};
    for (auto i = 0; i < 6; ++i) {
      auto &set = team[i];
      const auto [species, slot] = partial.species_slots[i];
      const auto &moves = partial.move_sets[slot];
      set.level = RandomBattlesData::random_set_data(species).level;
      set.species = species;
      for (auto m = 0; m < 4; ++m) {
        set.moves[m] = moves._data[m];
      }
    }
    return team;
  }

  PartialTeam randomTeam() {

    PartialTeam team{};
    auto n_pokemon = 0;

    prng.next(); // for type sample call

    std::array<int, 15> typeCount{};
    std::array<int, 6> weaknessCount{};
    int numMaxLevelPokemon{};

    using Arr = ArrayBasedVector<146>::Vector<Species>;

    Arr pokemonPool{RandomBattlesData::pokemonPool};
    Arr rejectedButNotInvalidPool{};

    while (n_pokemon < 6 && pokemonPool.size()) {
      auto species = sampleNoReplace(pokemonPool, prng);

      if (species == Species::Ditto && battleHasDitto) {
        continue;
      }

      bool skip = false;

      // types
      const auto types = get_types(species);
      for (const Type type : types) {
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
      if (species == Species::Ditto) {
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

  PartialSet randomSet(Species species) {
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

    ArrayBasedVector<RandomBattlesData::RandomSetEntry::max_moves>::Vector<Move>
        movePool{data.moves};
    movePool.resize(data.n_moves);
    while ((set_size < maxMoveCount) && movePool.size()) {
      set_size += set.insert(sampleNoReplace(movePool, prng));
    }

    assert((set_size == maxMoveCount) || (species == Species::Ditto));

    prng.shuffle(set._data);
    // sort before returning for fast comparison
    set.sort();

    return set;
  }

  PartialSet finishSet(const Species species, const PartialSet &set) {
    while (true) {
      const auto finished = randomSet(species);
      if (finished.contains(set)) {
        return finished;
      }
    }
  }

  // PartialTeam finishTeam(PartialTeam team = {}) {

  //   auto n_pokemon = 0;
  //   std::array<int, 15> typeCount{};
  //   std::array<int, 6> weaknessCount{};
  //   int numMaxLevelPokemon{};

  //   using Arr = ArrayBasedVector<146>::Vector<Species>;

  //   Arr pokemonPool{RandomBattlesData::pokemonPool};
  //   Arr rejectedButNotInvalidPool{};

  //   for (const auto& set : team) {
  //     const auto species = set.species;
  //     if (species != Species::None) {
  //       ++n_pokemon;
  //       const auto si = std::find(pokemonPool.begin(), pokemonPool.end(),
  //       species); assert(si != pokemonPool.end()); const auto _ =
  //       fastPop(pokemonPool, std::distance(pokemonPool.begin(), si)); const
  //       auto& types = get_types(species);
  //       typeCount[static_cast<uint8_t>(types[0])]++;
  //       if (types[0] != types[1]) {
  //         typeCount[static_cast<uint8_t>(types[1])]++;
  //       }
  //       const auto &w =
  //           RandomBattlesData::RANDOM_SET_DATA[static_cast<uint8_t>(species)]
  //               .weaknesses;
  //       }
  //       for (int i = 0; i < 6; ++i) {
  //         weaknessCount[i] += w[i];
  //       }
  //       if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1)
  //       {
  //         ++numMaxLevelPokemon;
  //       }
  //       if (species == Species::Ditto) {
  //         battleHasDitto = true;
  //       }
  //   }

  //   while (n_pokemon < 6 && pokemonPool.size()) {
  //     auto species = sampleNoReplace(pokemonPool, prng);

  //     if (species == Species::Ditto && battleHasDitto) {
  //       continue;
  //     }

  //     bool skip = false;

  //     // types
  //     const auto types = get_types(species);
  //     for (const Type type : types) {
  //       if (typeCount[static_cast<uint8_t>(type)] >= 2) {
  //         skip = true;
  //         break;
  //       }
  //     }
  //     if (skip) {
  //       rejectedButNotInvalidPool.push_back(species);
  //       continue;
  //     }

  //     // weakness
  //     const auto &w =
  //         RandomBattlesData::RANDOM_SET_DATA[static_cast<uint8_t>(species)]
  //             .weaknesses;
  //     for (int i = 0; i < 6; ++i) {
  //       if (!w[i]) {
  //         continue;
  //       }
  //       if (weaknessCount[i] >= 2) {
  //         skip = true;
  //         break;
  //       }
  //     }
  //     if (skip) {
  //       rejectedButNotInvalidPool.push_back(species);
  //       continue;
  //     }

  //     // lvl 100
  //     if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1) {
  //       skip = true;
  //     }
  //     if (skip) {
  //       rejectedButNotInvalidPool.push_back(species);
  //       continue;
  //     }

  //     // accept the set
  //     team.species_slots[n_pokemon] = {species, n_pokemon};
  //     team.move_sets[n_pokemon++] = randomSet(species);

  //     // update
  //     typeCount[static_cast<uint8_t>(types[0])]++;
  //     if (types[0] != types[1]) {
  //       typeCount[static_cast<uint8_t>(types[1])]++;
  //     }

  //     for (int i = 0; i < 6; ++i) {
  //       weaknessCount[i] += w[i];
  //     }
  //     if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1) {
  //       ++numMaxLevelPokemon;
  //     }
  //     if (species == Species::Ditto) {
  //       battleHasDitto = true;
  //     }
  //   }

  //   while (n_pokemon < 6 && rejectedButNotInvalidPool.size()) {
  //     const auto species = sampleNoReplace(rejectedButNotInvalidPool, prng);
  //     team.species_slots[n_pokemon] = {species, n_pokemon};
  //     team.move_sets[n_pokemon++] = randomSet(species);
  //   }

  //   team.sort();
  //   return team;

  // }
};

} // namespace RandomBattles
